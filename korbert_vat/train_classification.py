from pytorch_transformers.modeling_bert import BertConfig
from pytorch_transformers import optimization

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets import Datasets
from model import BertForEmotionClassification
from optim import layerwise_decay_optimizer

from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import time
import pickle
import utils
import logging
import argparse
import random
import os
import warnings

warnings.filterwarnings('ignore')
logger = utils.get_logger('BERT Classification')
logger.setLevel(logging.INFO)


class ClassificationBatchFunction:
    # batch function for pytorch dataloader
    def __init__(self, max_len, pad_idx, cls_idx, sep_idx):
        self.max_len = max_len
        self.pad_idx = pad_idx
        self.cls_idx = cls_idx
        self.sep_idx = sep_idx

    def __call__(self, batch):
        tokens, label = list(zip(*batch))

        # Get max length from batch
        max_len = min(self.max_len, max([len(i) for i in tokens]))
        tokens = torch.tensor([self.pad([self.cls_idx] + t + [self.sep_idx], max_len) for t in tokens])
        masks = torch.ones_like(tokens).masked_fill(tokens == self.pad_idx, 0)

        return tokens, masks, torch.tensor(label)

    def pad(self, sample, max_len):
        diff = max_len - len(sample)
        if diff > 0:
            sample += [self.pad_idx] * diff
        else:
            sample = sample[-max_len:]
        return sample


def train(args):
    set_seed(args)
    args.train_batch_size = args.train_batch_size*args.n_gpu 
    args.eval_batch_size = args.eval_batch_size*args.n_gpu 
    
    # Set device
    if args.device == 'cuda':
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        logger.info('use cuda')
    else:
        device = torch.device('cpu')
        logger.info('use cpu')

    # Set label list for classification
    if args.num_label == 'multi':
        label_list = ['공포', '놀람', '분노', '슬픔', '중립', '행복', '혐오']
    elif args.num_label == 'binary':
        label_list = ['긍정', '부정']
    logger.info('use {} labels for training'.format(len(label_list)))

    # Load pretrained model and model configuration
    pretrained_path = os.path.join('./pretrained_model/', args.pretrained_type)
    if args.pretrained_model_path is None:
        # Use pretrained bert model(etri/skt)
        pretrained_model_path = os.path.join(pretrained_path, 'pytorch_model.bin')
    else:
        # Use further-pretrained bert model
        pretrained_model_path = args.pretrained_model_path
    logger.info('Pretrain Model : {}'.format(pretrained_model_path))
    pretrained = torch.load(pretrained_model_path)
    
    # weight
    if args.pretrained_type == 'skt' and 'bert.' not in list(pretrained.keys())[0]:
        logger.info('modify parameter names')
        # Change parameter name for consistency
        new_keys_ = ['bert.' + k for k in pretrained.keys()]
        old_values_ = pretrained.values()
        pretrained = {k: v for k, v in zip(new_keys_, old_values_)}

    # bulid model
    bert_config = BertConfig(os.path.join(pretrained_path + '/bert_config.json'))
    bert_config.num_labels = len(label_list)
    model = BertForEmotionClassification(bert_config).to(device)
    

    # assigning weight
    model.load_state_dict(pretrained, strict=False)

    # Load Datasets
    tr_set = Datasets(file_path=args.train_data_path,
                      label_list=label_list,
                      pretrained_type=args.pretrained_type,
                      max_len=args.max_len)
    dev_set = Datasets(file_path=args.dev_data_path,
                       label_list=label_list,
                       pretrained_type=args.pretrained_type,
                       max_len=args.max_len)
    
    # Use custom batch function
    collate_fn = ClassificationBatchFunction(args.max_len, tr_set.pad_idx, tr_set.cls_idx, tr_set.sep_idx)
    
    tr_loader = DataLoader(dataset=tr_set,
                           batch_size=args.train_batch_size,
                           shuffle=True,
                           num_workers=8,
                           pin_memory=True,
                           collate_fn=collate_fn)


    dev_loader = DataLoader(dataset=dev_set,
                            batch_size=args.eval_batch_size,
                            num_workers=8,
                            pin_memory=True,
                            drop_last=False,
                            collate_fn=collate_fn)

    # optimizer
    optimizer = layerwise_decay_optimizer(model=model, lr=args.learning_rate, layerwise_decay=args.layerwise_decay)

    # lr scheduler
    t_total = len(tr_loader) // args.gradient_accumulation_steps * args.epochs 
    warmup_steps = int(t_total * args.warmup_percent)
    logger.info('total training steps : {}, lr warmup steps : {}'.format(t_total, warmup_steps))
    # Use gradual warmup and linear decay
    scheduler = optimization.WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)

    # for low-precision training
    if args.fp16:
        try:
            from apex import amp
            logger.info('Use fp16')
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level, verbosity=0)

    # tensorboard setting
    save_path = "./model_saved_finetuning/lr{},batch{},total{},warmup{},len{},{}".format(
        args.learning_rate, args.train_batch_size * args.gradient_accumulation_steps, t_total,
        args.warmup_percent, args.max_len, args.pretrained_type)

    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    writer = SummaryWriter(save_path)

    # Save best model results with resultwriter
    result_writer = utils.ResultWriter("./model_saved_finetuning/results.csv")
    model.zero_grad()

    best_val_loss = 1e+9
    global_step = 0
    
    train_loss, train_acc, train_f1 = 0, 0, 0
    logging_loss, logging_acc, logging_f1 = 0, 0, 0

    logger.info('***** Training starts *****')
    total_result = []
    for epoch in tqdm(range(args.epochs), desc='epochs'):
        for step, batch in tqdm(enumerate(tr_loader), desc='steps', total=len(tr_loader)):
            model.train()
            x_train, mask_train, y_train = map(lambda x: x.to(device), batch)

            inputs = {
                'input_ids': x_train,
                'attention_mask': mask_train,
                'classification_label': y_train,
            }

            output, loss = model(**inputs)
            y_max = output.max(dim=1)[1]

            cr = classification_report(y_train.tolist(),
                                       y_max.tolist(),
                                       labels=list(range(len(label_list))),
                                       target_names=label_list,
                                       output_dict=True)
            # Get accuracy(micro f1)
            if 'micro avg' not in cr.keys():
                batch_acc = list(cr.items())[len(label_list)][1]
            else:
                # If at least one of labels does not exists in mini-batch, use micro average instead
                batch_acc = cr['micro avg']['f1-score']
            # macro f1
            batch_macro_f1 = cr['macro avg']['f1-score']

            # accumulate measures
            grad_accu = args.gradient_accumulation_steps
            if grad_accu > 1:
                loss /= grad_accu
                batch_acc /= grad_accu
                batch_macro_f1 /= grad_accu

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                if args.n_gpu > 1:
                    loss = loss.mean()
                    loss.backward()
                else:
                    loss.backward()

            train_loss += loss.item()
            train_acc += batch_acc
            train_f1 += batch_macro_f1

            if (global_step + 1) % grad_accu == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.grad_clip_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1
                
                if global_step % args.logging_step == 0:
                    acc_ = (train_acc - logging_acc) / args.logging_step
                    f1_ = (train_f1 - logging_f1) / args.logging_step
                    loss_ = (train_loss - logging_loss) / args.logging_step
                    writer.add_scalars('loss', {'train': loss_}, global_step)
                    writer.add_scalars('acc', {'train': acc_}, global_step)
                    writer.add_scalars('macro_f1', {'train': f1_}, global_step)

                    logger.info('[{}/{}], trn loss : {:.3f}, trn acc : {:.3f}, macro f1 : {:.3f}'.format(
                        global_step, t_total, loss_, acc_, f1_
                    ))
                    logging_acc, logging_f1, logging_loss = train_acc, train_f1, train_loss

                    # Get f1 score for each label
                    f1_results = [(l, r['f1-score']) for i, (l, r) in enumerate(cr.items()) if i < len(label_list)]
                    f1_log = "\n".join(["{} : {}".format(l, f) for l, f in f1_results])
                    logger.info("\n\n***f1-score***\n" + f1_log + "\n\n***confusion matrix***\n{}".format(
                        confusion_matrix(y_train.tolist(), y_max.tolist())))

        # Validation
        val_loss, val_acc, val_macro_f1, _ = evaluate(args, dev_loader, model, device)
        val_result = '[{}/{}] val loss : {:.3f}, val acc : {:.3f}. val macro f1 : {:.3f}'.format(
            global_step, t_total, val_loss, val_acc, val_macro_f1
        )

        writer.add_scalars('loss', {'val': val_loss}, global_step)
        writer.add_scalars('acc', {'val': val_acc}, global_step)
        writer.add_scalars('macro_f1', {'val': val_macro_f1}, global_step)
        logger.info(val_result)
        total_result.append(val_result)

        if val_loss < best_val_loss:
            # Save model checkpoints
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.bin'))
            torch.save(args, os.path.join(save_path, 'training_args.bin'))
            logger.info('Saving model checkpoint to %s', save_path)
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_val_macro_f1 = val_macro_f1

    # Save results in 'model_saved_finetuning/results.csv'
    results = {
        'val_loss': best_val_loss,
        'val_acc': best_val_acc,
        'val_macro_f1' : best_val_macro_f1,
        'save_dir': save_path,
        'pretrained_path': pretrained_path,
    }
    result_writer.update(args, **results)
    return global_step, loss_, acc_, best_val_loss, best_val_acc, total_result


def evaluate(args, dataloader, model, device, objective='classification'):

    if args.num_label == 'multi':
        label_list = ['공포', '놀람', '분노', '슬픔', '중립', '행복', '혐오']
    elif args.num_label == 'binary':
        label_list = ['긍정', '부정']

    val_loss, val_acc, val_f1 = 0, 0, 0
    total_y = []
    total_y_hat = []

    for val_step, batch in enumerate(dataloader):
        model.eval()

        x_dev, mask_dev, y_dev = map(lambda x: x.to(device), batch)
        total_y += y_dev.tolist()

        inputs = {
            'input_ids': x_dev,
            'attention_mask': mask_dev,
            'classification_label': y_dev,
        }
        with torch.no_grad():
            output, loss = model(**inputs)
            if args.n_gpu > 1:
                loss = loss.mean()
            y_max = output.max(dim=1)[1]
            total_y_hat += y_max.tolist()
            val_loss += loss.item()

    # f1-score 계산
    dev_cr = classification_report(total_y,
                                   total_y_hat,
                                   labels=list(range(len(label_list))),
                                   target_names=label_list,
                                   output_dict=True)

    # Get accuracy(micro f1)
    if 'micro avg' not in dev_cr.keys():
        val_acc = list(dev_cr.items())[len(label_list)][1]
    else:
        # If at least one of labels does not exists in mini-batch, use micro average instead
        val_acc = dev_cr['micro avg']['f1-score']
    # macro f1
    val_macro_f1 = dev_cr['macro avg']['f1-score']

    logger.info('***** Evaluation Results *****')
    f1_results = [(l, r['f1-score']) for i, (l, r) in enumerate(dev_cr.items()) if i < len(label_list)]
    f1_log = "\n".join(["{} : {}".format(l, f) for l, f in f1_results])
    cm = confusion_matrix(total_y, total_y_hat)
    logger.info("\n***f1-score***\n" + f1_log + "\n***confusion matrix***\n{}".format(cm))

    val_loss /= (val_step + 1)
    return val_loss, val_acc, val_macro_f1, (total_y_hat, cm)


def set_seed(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def main():
    parser = argparse.ArgumentParser()

    # Pretrained model Parameters
    parser.add_argument("--pretrained_type", default='etri', type=str,
                        help="type of pretrained model (skt, etri)")
    parser.add_argument("--pretrained_model_path", required=False,
                        help="path of pretrained model (If you wnat to use further-pretrained model)")

    # Train Parameters
    parser.add_argument("--train_batch_size", default=100, type=int,
                        help="batch size")
    parser.add_argument("--eval_batch_size", default=100, type=int,
                        help="batch size for validation")
    parser.add_argument("--layerwise_decay", action="store_true",
                        help="Whether to use layerwise decay")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam")
    parser.add_argument("--epochs", default=50, type=int,
                        help="total epochs")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="gradient accumulation steps for large batch training")
    parser.add_argument("--warmup_percent", default=0.1, type=float,
                        help="gradient warmup percentage")
    parser.add_argument("--grad_clip_norm", default=1.0, type=float,
                        help="batch size")

    # Other Parameters
    parser.add_argument("--logging_step", default=2, type=int,
                        help="logging step for training loss and acc")
    parser.add_argument("--device", default='cuda', type=str,
                        help="Whether to use cpu or cuda")
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use nvidia mixed precision training")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--seed", default=0, type=int,
                        help="Random seed(default=0)")
    parser.add_argument("--n_gpu", default=1, type=int,
                        help="n_gpu")
    # Data Parameters
    parser.add_argument("--train_data_path", default='../split_datasets_vat/train.pkl', type=str,
                        help="train data path")
    parser.add_argument("--dev_data_path", default='../split_datasets_vat/dev.pkl', type=str,
                        help="dev data path")
    parser.add_argument("--num_label", default='multi', type=str,
                        help="Number of labels in datastes(binary or multi)")
    parser.add_argument("--max_len", default=64, type=int,
                        help="Maximum sequence length")

    args = parser.parse_args()
    set_seed(args)

    t = time.time()
    global_step, train_loss, train_acc, best_val_loss, best_val_acc, total_result = train(args)
    elapsed = time.time() - t

    logger.info('***** Training done *****')
    logger.info('total validation results : \n'+'\n'.join(total_result))
    logger.info('elapsed time: %.3f Hours' % (elapsed / 3600.))
    logger.info('best acc in test: %.4f' % best_val_acc)
    logger.info('best loss in test: %.4f' % best_val_loss)


if __name__ == '__main__':
    main()
