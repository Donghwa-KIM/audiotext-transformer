from pytorch_transformers.modeling_bert import BertConfig
from pytorch_transformers import optimization

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets import Datasets
from model import BertForMLM
from optim import layerwise_decay_optimizer

from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import time
import pickle
import utils
import logging
import argparse
import random
import copy
import os
import warnings

warnings.filterwarnings('ignore')
logger = utils.get_logger('BERT MLM')
logger.setLevel(logging.INFO)


class MLMBatchFunction:
    # batch function for pytorch dataloader
    def __init__(self, max_len, vocab):
        self.max_len = max_len
        self.vocab = vocab
        self.pad_idx = self.vocab['[PAD]']
        self.cls_idx = self.vocab['[CLS]']
        self.sep_idx = self.vocab['[SEP]']
        self.mask_idx = self.vocab['[MASK]']

    def __call__(self, batch):
        # list(zip(*[tr_set[0],tr_set[1]]))
        tokens, label = list(zip(*batch))

        # Get max length from batch
        max_len = min(self.max_len, max([len(i) for i in tokens]))

        # Use unmasked tokens as vocab labels for MLM
        masked_tokens, masked_idcs = list(zip(*[self.make_masked_input(t) for t in copy.deepcopy(tokens)]))
        masked_tokens = torch.tensor([self.pad([self.cls_idx] + t + [self.sep_idx], max_len, self.pad_idx) for t in masked_tokens])
        masked_idcs = torch.tensor([self.pad([self.pad_idx] + t + [self.pad_idx], max_len, 0) for t in masked_idcs])
        
        label = torch.tensor([self.pad([-1] + t + [-1], max_len, -1) for t in tokens])
        label = label.masked_fill(masked_idcs != 1, -1)

        masks = torch.ones_like(masked_tokens).masked_fill(masked_tokens == self.pad_idx, 0)

        return masked_tokens, label, masks

    def pad(self, sample, max_len, pad):
        diff = max_len - len(sample)
        if diff > 0:
            sample += [pad] * diff
        else:
            sample = [sample[0]] + sample[-max_len+1:]
        return sample
    
    def make_masked_input(self, sample):
        if len(sample) > self.max_len - 2:
            sample = sample[:self.max_len - 2]
            
        masked_idcs = []
        for i, token in enumerate(sample):
            # sentence must have at least one mask
            prob = random.random()
            # mask token with 15%
            if prob < 0.15:
                masked_idcs.append(1)
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    sample[i] = self.mask_idx

                # 10% randomly change token to random token
                elif prob < 0.9:
                    special_tokens = [self.pad_idx, self.mask_idx, self.sep_idx, self.cls_idx]
                    while True:
                        r = random.choice(range(len(self.vocab)))
                        if r not in special_tokens:
                            break
                    sample[i] = r
            else:
                masked_idcs.append(0)

        return sample, masked_idcs

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

    if args.pretrained_type == 'skt' and 'bert.' not in list(pretrained.keys())[0]:
        logger.info('modify parameter names')
        # Change parameter name for consistency
        new_keys_ = ['bert.' + k for k in pretrained.keys()]
        old_values_ = pretrained.values()
        pretrained = {k: v for k, v in zip(new_keys_, old_values_)}

    bert_config = BertConfig(os.path.join(pretrained_path+ '/bert_config.json'))
    model = BertForMLM(bert_config).to(device)
    model.load_state_dict(pretrained, strict=False)

    # Load Datasets
    tr_set = Datasets(file_path=args.train_data_path,
                      pretrained_type=args.pretrained_type,
                      max_len=args.max_len)
    
    # Use custom batch function
    collate_fn = MLMBatchFunction(args.max_len, tr_set.vocab)
    
    tr_loader = DataLoader(dataset=tr_set,
                           batch_size=args.train_batch_size,
                           shuffle=True,
                           num_workers=8,
                           pin_memory=True,
                           drop_last=True,
                           collate_fn=collate_fn)

    if args.do_eval:
        dev_set = Datasets(file_path=args.dev_data_path,
                        pretrained_type=args.pretrained_type,
                        max_len=args.max_len)
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
    # Use gradual warmup and cosine decay
    scheduler = optimization.WarmupCosineWithHardRestartsSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)

    # for low-precision training
    if args.fp16:
        try:
            from apex import amp
            logger.info('Use fp16')
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level, verbosity=0)

    # tensorboard setting
    save_path = "./model_saved_pretrain/lr{},batch{},total{},warmup{},len{},{}".format(
        args.learning_rate, args.train_batch_size * args.gradient_accumulation_steps,
        t_total, args.warmup_percent, args.max_len, args.pretrained_type)

    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    writer = SummaryWriter(save_path)

    # Save best model results with resultwriter
    # just csv out
    result_writer = utils.ResultWriter("./model_saved_pretrain/results.csv")
    model.zero_grad()

    best_val_loss = 1e+9
    best_val_acc = 0
    global_step = 0
    
    train_loss, train_acc = 0, 0
    val_loss, val_acc = 0, 0
    logging_loss, logging_acc = 0, 0

    logger.info('***** Training starts *****')
    total_result = []
    for epoch in tqdm(range(args.epochs), desc='epochs'):
        for step, batch in tqdm(enumerate(tr_loader), desc='steps', total=len(tr_loader)):
            model.train()
            x_train, y_train, mask_train = map(lambda x: x.to(device), batch)

            inputs = {
                'input_ids': x_train,
                'attention_mask': mask_train,
                'masked_lm_labels': y_train,
            }

            output, loss = model(**inputs)
            y_max = output.max(dim=2)[1]
            
            # Get accuracy for maked tokens
            total_length = torch.ones_like(y_train).masked_fill(y_train == -1, 0).sum().item()
            total_sum = torch.zeros_like(y_max).masked_fill(y_max == y_train, 1).sum().item()
            batch_acc = total_sum / total_length

            # accumulate measures
            grad_accu = args.gradient_accumulation_steps
            if grad_accu > 1:
                loss /= grad_accu
                batch_acc /= grad_accu

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
            if (step + 1) % grad_accu == 0:
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
                    loss_ = (train_loss - logging_loss) / args.logging_step
                    writer.add_scalars('loss', {'train': loss_}, global_step)
                    writer.add_scalars('acc', {'train': acc_}, global_step)
                    writer.add_scalars('lr', {'lr': scheduler.get_lr()[0]}, global_step)
                    
                    logger.info('[{}/{}], trn loss : {:.3f}, trn acc : {:.3f}'.format(
                        global_step, t_total, loss_, acc_
                    ))
                    
                    logging_acc, logging_loss = train_acc, train_loss
        
        if args.do_eval:
            # Validation
            val_loss, val_acc= evaluate(args, dev_loader, model, device)
            val_result = '[{}/{}] val loss : {:.3f}, val acc : {:.3f}'.format(
                global_step, t_total, val_loss, val_acc
            )
            logger.info(val_result)
            total_result.append(val_result)

        if val_loss <= best_val_loss:
            # Save model checkpoints
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.bin'))
            torch.save(args, os.path.join(save_path, 'training_args.bin'))
            logger.info('Saving model checkpoint to %s', save_path)
            best_val_loss = val_loss
            best_val_acc = val_acc
        
        if (epoch + 1) % args.saving_step == 0:
            torch.save(model.state_dict(), os.path.join(save_path, 'epoch{}_model.bin'.format(epoch+1)))
            # Save results in 'model_saved_pretrain/results.csv'
            results = {
                'train_loss': loss_,
                'train_acc': acc_, 
                'val_loss': best_val_loss,
                'val_acc': best_val_acc,
                'save_dir': save_path,
                'global_step': global_step,
            }
            result_writer.update(args, **results)
    
    return global_step, loss_, acc_, best_val_loss, best_val_acc, total_result


def evaluate(args, dataloader, model, device):

    val_loss, val_acc, val_f1 = 0, 0, 0
    total_y = []
    total_y_hat = []

    for val_step, batch in enumerate(dataloader):
        model.eval()

        x_dev, y_dev, mask_dev  = map(lambda x: x.to(device), batch)
        total_y += y_dev.tolist()

        inputs = {
            'input_ids': x_dev,
            'attention_mask': mask_dev,
            'masked_lm_labels': y_dev,
        }
        with torch.no_grad():
            output, loss = model(**inputs)
            if args.n_gpu > 1:
                loss = loss.mean()
            y_max = output.max(dim=2)[1]
            total_y_hat += y_max.tolist()
            val_loss += loss.item()
    
    # Make lists unnested and get accuracy for masked tokens
    total_y, total_y_hat = [torch.tensor([x for sublist in y for x in sublist]) for y in (total_y, total_y_hat)]
    total_length = torch.ones_like(total_y).masked_fill(total_y == -1, 0).sum().item()
    total_sum = torch.zeros_like(total_y_hat).masked_fill(total_y_hat == total_y, 1).sum().item()
    
    val_acc = total_sum / total_length
    val_loss /= (val_step + 1)
    return val_loss, val_acc


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
    parser.add_argument("--train_batch_size", default=64, type=int,
                        help="batch size")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="batch size for validation")
    parser.add_argument("--layerwise_decay", action="store_true",
                        help="Whether to use layerwise decay")
    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for Adam")
    parser.add_argument("--epochs", default=70, type=int,
                        help="total epochs")
    parser.add_argument("--gradient_accumulation_steps", default=4, type=int,
                        help="gradient accumulation steps for large batch training")
    parser.add_argument("--warmup_percent", default=0.005, type=float,
                        help="gradient warmup percentage")
    parser.add_argument("--grad_clip_norm", default=1.0, type=float,
                        help="batch size")

    # Other Parameters
    parser.add_argument("--logging_step", default=1, type=int,
                        help="logging step for training loss and acc")
    parser.add_argument("--saving_step", default=5, type=int,
                        help="epoch for saving temporarily")
    parser.add_argument("--do_eval", action="store_true",
                        help="do evaluation step for maked language model")
    parser.add_argument("--device", default='cuda', type=str,
                        help="Whether to use cpu or cuda")
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use nvidia mixed precision training")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--seed", default=0, type=int,
                        help="Random seed(default=0)")
    parser.add_argument("--n_gpu", default=1, type=int,
                        help="n_gpu")
    # Data Parameters
    parser.add_argument("--train_data_path", default='../split_datasets_1channel/train.pkl', type=str,
                        help="train data path")
    parser.add_argument("--dev_data_path", default='../split_datasets_1channel/dev.pkl', type=str,
                        help="dev data path")
    parser.add_argument("--max_len", default=90, type=int,
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
