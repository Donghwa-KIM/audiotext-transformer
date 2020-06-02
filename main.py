import argparse
import logging
import random
import os
from tqdm import tqdm, trange
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import itertools



# audio
import librosa
import torchaudio

# text
from pytorch_transformers.modeling_bert import BertPreTrainedModel
from pytorch_transformers.modeling_bert import BertConfig, BertModel

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import (
    DataLoader,
    Dataset,
    RandomSampler,
    SequentialSampler,
    DistributedSampler,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau

from datasets import DataSets
from model import MULTModel
from utils import *
from sklearn.metrics import classification_report, confusion_matrix

from warmup_scheduler import GradualWarmupScheduler

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
        
class AudioTextBatchFunction:
    
    # batch function for pytorch dataloader
    def __init__(self,
                 args,
                 pad_idx, cls_idx, sep_idx,
                 bert_args,
                 num_label_from_bert,
                 device = 'cpu'
                 ):
        self.only_audio = args.do_audio
        self.device = device

        # related to audio--------------------
        self.max_len_a = args.max_len_for_audio
        self.n_mfcc = args.n_mfcc
        self.n_fft_size = args.n_fft_size
        self.sample_lr = args.sample_rate
        self.resample_lr = args.resample_rate

        self.audio2mfcc = torchaudio.transforms.MFCC(sample_rate=self.resample_lr,
                                        n_mfcc= self.n_mfcc,
                                        log_mels=False,
                                        melkwargs = {'n_fft': self.n_fft_size}).to(self.device)
        
        if not self.only_audio:
            # related to text--------------------
            self.max_len_t = bert_args.max_len
            self.pad_idx = pad_idx
            self.cls_idx = cls_idx
            self.sep_idx = sep_idx


            self.bert_config = BertConfig(args.bert_config_path)
            self.bert_config.num_labels = num_label_from_bert

            self.model = BertForTextRepresentation(self.bert_config).to(self.device)
            pretrained_weights = torch.load(args.bert_model_path
                                                , map_location=torch.device(self.device))
            self.model.load_state_dict(pretrained_weights, strict=False)
            self.model.eval()


        
    def __call__(self, batch):
        audio, texts, label = list(zip(*batch))

        if not self.only_audio:
            # Get max length from batch
            max_len = min(self.max_len_t, max([len(i) for i in texts]))
            texts = torch.tensor([self.pad_with_text([self.cls_idx] + text + [self.sep_idx], max_len) for text in texts])
            masks = torch.ones_like(texts).masked_fill(texts == self.pad_idx, 0)

            with torch.no_grad():
                # text_emb = last layer
                text_emb, cls_token = self.model(**{'input_ids': texts.to(self.device),
                                                   'attention_mask': masks.to(self.device)})
            
                audio_emb, audio_mask = self.pad_with_mfcc(audio)
            
            return audio_emb, audio_mask, text_emb, torch.tensor(label)
        else:
            audio_emb, audio_mask = self.pad_with_mfcc(audio)
            return audio_emb, audio_mask, None, torch.tensor(label)



    def pad_with_text(self, sample, max_len):
        diff = max_len - len(sample)
        if diff > 0:
            sample += [self.pad_idx] * diff
        else:
            sample = sample[-max_len:]
        return sample

    def pad_with_mfcc(self, audios): 
        max_len_batch = min(self.max_len_a, max([len(a) for a in audios]))
        audio_array = torch.zeros(len(audios), self.n_mfcc, max_len_batch).fill_(float('-inf')).to(self.device)
        for ix, audio in enumerate(audios):
            audio_ = librosa.core.resample(audio, self.sample_lr, self.resample_lr)
            audio_ = torch.tensor(self.trimmer(audio_))
            mfcc = self.audio2mfcc(audio_.to(self.device))
            sel_ix = min(mfcc.shape[1], max_len_batch)
            audio_array[ix,:,:sel_ix] = mfcc[:,:sel_ix]
        # (bat, n_mfcc, seq) -> (bat, seq, n_mfcc)
        padded_array = audio_array.transpose(2,1)
        
        # key masking
        # (batch, seq)
        key_mask = padded_array[:,:,0]
        key_mask = key_mask.masked_fill(key_mask != float('-inf'), 0).masked_fill(key_mask == float('-inf'),1).bool()
        
        # -inf -> 0.0
        padded_array = padded_array.masked_fill(padded_array == float('-inf'), float(0))
        return padded_array, key_mask
    
    def trimmer(self, audio):
        fwd_audio = []
        fwd_init = np.float32(0)
        for a in audio:
            if fwd_init!=np.float32(a):
                fwd_audio.append(a)

        bwd_init = np.float32(0)
        bwd_audio =[]
        for a in fwd_audio[::-1]:
            if bwd_init!=np.float32(a):
                bwd_audio.append(a)
        return bwd_audio[::-1]
            


class BertForTextRepresentation(BertPreTrainedModel):

    def __init__(self, config):
        super(BertForTextRepresentation, self).__init__(config)
        self.bert = BertModel(config)
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.fc = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                masked_lm_labels=None, classification_label=None):
        bert_output = self.bert(input_ids=input_ids,
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                token_type_ids=token_type_ids,
                                head_mask=head_mask)
        return bert_output


    
    
    



def train(args, bert_args, label_list, train_dataset, eval_dataset, model):
    sampler = (
        RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    )
    
    collate_fn = AudioTextBatchFunction(args= args,
                                    pad_idx = train_dataset.pad_idx,
                                    cls_idx = train_dataset.cls_idx, 
                                    sep_idx = train_dataset.sep_idx,
                                    bert_args = bert_args,
                                    num_label_from_bert = len(label_list),
                                    device ='cpu'
                                   )


    args.batch_size = args.batch_size * max(1, args.n_gpu)
    
    train_loader = DataLoader(train_dataset, 
                              sampler=sampler,
                              batch_size=args.batch_size,
                              collate_fn=collate_fn,
                              num_workers=12)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # total num_batches
    t_total = len(train_loader) * args.num_epochs
    args.warmup_step = int(args.warmup_percent * t_total)

    if args.warmup_step != 0:
        # decay learning rate, related to a validation
        scheduler_plateau = ReduceLROnPlateau(
            optimizer, "min", patience=args.when, factor=0.1, verbose=True
        )
        scheduler = GradualWarmupScheduler(
            optimizer, 1, args.warmup_step, after_scheduler=scheduler_plateau
        )
    else:
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", patience=args.when, factor=0.1, verbose=True
        )
    loss_fct = torch.nn.CrossEntropyLoss()

    # Train!
    logger.info(f"***** Running {args.cls_model_name}-{args.merge_how} Transformer *****")
    logger.info("  Num Epochs = %d", args.num_epochs)
    logger.info(
        "  Total train batch size = %d",
        args.batch_size * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Warmup Steps = %d", args.warmup_step)

    global_step = 0
    steps_trained_in_current_epoch = 0
    # cum_loss, current loss
    tr_loss, logging_loss = 0.0, 0.0
    best_val_loss = 1e8
    

    model.zero_grad()
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    train_iterator = trange(
        0, int(args.num_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
    )
    
    set_seed(args)
    
    
    args.params_name = '-'.join([k+'='+ str(vars(args)[k]) for k in vars(args) if k not in ['warmup_step',
                                                                                            'cls_model_name','device','n_gpu','attn_mask','do_audio',
                                                                                            'model','data_name','data_path',
                                                                                            'bert_config_path','bert_model_path','bert_args_path','relu_dropout',
                                                                                            'res_dropout','layers','d_out','num_heads','batch_size','clip',
                                                                                            'optim', 'max_len_for_audio','sample_rate','logging_steps','seed',
                                                                                            'no_cuda','name','hidden_size_for_bert','max_len_for_text',
                                                                                            'warmup_percent','when','batch_chunk','local_rank','num_epochs'
                                                                      ]])

    if not os.path.isdir(f"{args.cls_model_name}/{args.params_name}"):
        os.makedirs(f"{args.cls_model_name}/{args.params_name}")
        
    result_writer = ResultWriter(f"./{args.cls_model_name}/train_dev_results.csv")

        
    for _ in train_iterator:
        epoch_iterator = tqdm(
            train_loader, desc="Iteration", disable=args.local_rank not in [-1, 0],
        )
        for step, (audios, audio_mask, texts, labels) in enumerate(epoch_iterator):
            
            model.train()
            
            if not args.do_audio:
                audios, audio_mask, texts = list(map(lambda x: x.to(args.device), [audios, audio_mask, texts]))
            else:
                audios, audio_mask = list(map(lambda x: x.to(args.device), [audios, audio_mask]))

            labels = labels.squeeze(-1).long().to(args.device)
            
            inputs = {'x_audio': audios,
                      'x_text': texts,
                      'a_mask': audio_mask}    
            
            
            preds, hidden = model(**inputs)

            loss = loss_fct(preds, labels.view(-1))
            
            if args.n_gpu > 1:
                loss = loss.mean()
            loss.backward()
            
            tr_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            model.zero_grad()
            global_step += 1

            if args.local_rank in [-1, 0] and global_step % args.logging_steps == 0:
                logger.info("  train loss : %.3f", (tr_loss - logging_loss) / args.logging_steps)
                logging_loss = tr_loss

        if args.local_rank in [-1, 0]:
            val_loss, val_acc, val_macro_f1, _ = evaluate(args, label_list, eval_dataset, model, loss_fct, collate_fn)
            val_result = '[{}/{}] val loss : {:.3f}, val acc : {:.3f}. val macro f1 : {:.3f}'.format(
            global_step, t_total, val_loss, val_acc, val_macro_f1
        )
            logger.info(val_result)

            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_val_macro_f1 = val_macro_f1
                
                
                
                torch.save(model, f"{args.cls_model_name}/{args.params_name}/model.pt")
                torch.save(args, f"{args.cls_model_name}/{args.params_name}/args.pt")
                logger.info(f"  Saved {args.cls_model_name}-{args.params_name}")

            logger.info("  val loss : %.3f", val_loss)
            logger.info("  best_val loss : %.3f", best_val_loss)
    
    results = {
        'val_loss': best_val_loss,
        'val_acc': best_val_acc,
        'val_macro_f1' : best_val_macro_f1
    }
    result_writer.update(args, **results)

def evaluate(args, label_list, eval_dataset, model, loss_fct, collate_fn):

    model.eval()
    sampler = SequentialSampler(eval_dataset)
    eval_loader = DataLoader(eval_dataset,
                             sampler=sampler, 
                             batch_size=args.batch_size,
                             collate_fn=collate_fn,
                             num_workers=12)

    val_loss, val_acc, val_f1 = 0, 0, 0
    total_y = []
    total_y_hat = []
    
    for val_step, (audios, audio_mask, texts, labels) in enumerate(tqdm(eval_loader, desc="Evaluating")):
        with torch.no_grad():
            
            if not args.do_audio:
                audios, audio_mask, texts = list(map(lambda x: x.to(args.device), [audios, audio_mask, texts]))
            else:
                audios, audio_mask = list(map(lambda x: x.to(args.device), [audios, audio_mask]))
            
            labels = labels.squeeze(-1).long().to(args.device)
            total_y += labels.tolist()
            
            inputs = {'x_audio': audios,
                      'x_text': texts,
                      'a_mask': audio_mask}   
            
            preds, hidden = model(**inputs)
            loss = loss_fct(preds, labels.view(-1))
            
            
            
            # max => out: (value, index)
            y_max = preds.max(dim=1)[1]
            total_y_hat += y_max.tolist()
            val_loss += loss.item()
            
    # f1-score 계산
    cr = classification_report(total_y,
                                   total_y_hat,
                                   labels=list(range(len(label_list))),
                                   target_names=label_list,
                                   output_dict=True)
    # Get accuracy(micro f1)
    if 'micro avg' not in cr.keys():
        val_acc = list(cr.items())[-1][1]['f1-score']
    else:
        # If at least one of labels does not exists in mini-batch, use micro average instead
        val_acc = cr['micro avg']['f1-score']
    # macro f1
    val_macro_f1 = cr['macro avg']['f1-score']

    logger.info('***** Evaluation Results *****')
    f1_results = [(l, r['f1-score']) for i, (l, r) in enumerate(cr.items()) if i < len(label_list)]
    f1_log = "\n".join(["{} : {}".format(l, f) for l, f in f1_results])
    cm = confusion_matrix(total_y, total_y_hat)
    logger.info("\n***f1-score***\n" + f1_log + "\n***confusion matrix***\n{}".format(cm))
    
    val_loss /= (val_step + 1)
    
    return val_loss, val_acc, val_macro_f1, (total_y_hat, cm, {l : f for l, f in f1_results})


def draw_cm(args, label_list, cm):
    fig, ax = plt.subplots(figsize=(9, 8))
    font_p = "./font/HANBatang.ttf"
    fontprop = fm.FontProperties(fname=font_p, size=15)

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(label_list))

    ax = plt.gca()
    plt.xticks(tick_marks)
    ax.set_xticklabels(label_list, fontproperties=fontprop)
    plt.yticks(tick_marks)
    ax.set_yticklabels(label_list, fontproperties=fontprop)
    
    plt.xlim(-0.5, len(label_list)-0.5)
    plt.ylim(-0.5, len(label_list)-0.5)
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=12)
    plt.tight_layout()
    plt.ylabel('True', fontsize=12)
    plt.xlabel('Predict', fontsize=12)
    plt.savefig(f'{args.cls_model_name}/{args.params_name}/test_result.png', dpi=300)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--do_audio", action="store_true",  help="whether to use of only audio (Transformer, etc.)"
    )
    
    parser.add_argument(
        "--merge_how", type=str, default="average", help="how to merge the sequence for a modal"
    )
    parser.add_argument(
        "--model", type=str, default="MulT", help="name of the model to use (Transformer, etc.)"
    )
    parser.add_argument(
        "--data_name", type=str, default="vat", help="dataset to use (default: vat)"
    )
    parser.add_argument(
        "--data_path", type=str, default="../../split_dataset_1channel", help="path for storing the dataset"
    )
    parser.add_argument(
        "--bert_config_path", type=str, default="./korbert_vat/best_model/bert_config.json", help="bert_config_path"
    )
    parser.add_argument(
        "--bert_args_path", type=str, default="./korbert_vat/best_model/training_args.bin", help="bert_args_path"
    )
    parser.add_argument(
        "--bert_model_path", type=str, default="./korbert_vat/best_model/best_model.bin", 
        help="bert_model_path (pretrained & finetuned by a sentiment task)"
    )


    # Dropouts
    parser.add_argument("--attn_dropout", type=float, default=0.1, help="attention dropout")
    parser.add_argument(
        "--attn_dropout_a", type=float, default=0.0, help="attention dropout (for audio)"
    )
    parser.add_argument(
        "--attn_dropout_v", type=float, default=0.0, help="attention dropout (for visual)"
    )
    parser.add_argument("--relu_dropout", type=float, default=0.1, help="relu dropout")
    parser.add_argument("--embed_dropout", type=float, default=0.25, help="embedding dropout")
    parser.add_argument("--res_dropout", type=float, default=0.1, help="residual block dropout")
    parser.add_argument("--out_dropout", type=float, default=0.0, help="output layer dropout")

    # Architecture
    parser.add_argument(
        "--layers", type=int, default=4, help="number of layers in the network (default: 5)"
    )
    parser.add_argument(
        "--d_model", type=int, default=30, help="dimension of layers in the network (default: 30)"
    )
    parser.add_argument(
        "--d_out", type=int, default=7, help="dimension of target dimension in the network (default: 7 for multi)"
    )

    parser.add_argument(
        "--num_heads",
        type=int,
        default=8,
        help="number of heads for the transformer network (default: 5)",
    )
    parser.add_argument(
        "--attn_mask",
        action="store_false",
        help="use attention mask for Transformer (default: true)",
    )

    # Tuning
    parser.add_argument(
        "--batch_size", type=int, default=64, metavar="N", help="batch size (default: 24)"
    )
    parser.add_argument(
        "--clip", type=float, default=0.8, help="gradient clip value (default: 0.8)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="initial learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--optim", type=str, default="Adam", help="optimizer to use (default: Adam)"
    )
    parser.add_argument("--num_epochs", type=int, default=40, help="number of epochs (default: 40)")
    parser.add_argument(
        "--when", type=int, default=10, help="when to decay learning rate (default: 20)"
    )
    parser.add_argument(
        "--batch_chunk", type=int, default=1, help="number of chunks per batch (default: 1)"
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training")
    parser.add_argument(
        "--warmup_percent", default=0.1, type=float, help="Linear warmup over warmup_percent."
    )

    parser.add_argument("--max_len_for_text", default=64, type=int,
                            help="Maximum sequence length for text")
    parser.add_argument("--hidden_size_for_bert", default=768, type=int,
                            help="hidden size used to a fine-tuned BERT")
    parser.add_argument("--max_len_for_audio", default=400, type=int,
                            help="Maximum sequence length for audio")
    parser.add_argument("--sample_rate", default=48000, type=int,
                            help="sampling rate for audio")
    parser.add_argument("--resample_rate", default= 16000, type=int,
                            help="resampling rate to reduce audio sequence")
    parser.add_argument("--n_fft_size", default=400, type=int,
                            help="time widnow for fourier transform")
    parser.add_argument("--n_mfcc", default=40, type=int,
                            help="low frequency range (from 0 to n_mfcc)")

    # Logistics
    parser.add_argument(
        "--logging_steps", type=int, default=30, help="frequency of result logging (default: 30)"
    )
    parser.add_argument("--seed", type=int, default=1234, help="random seed")
    parser.add_argument("--no_cuda", action="store_true", help="do not use cuda")
    parser.add_argument(
        "--name", type=str, default="mult", help='name of the trial (default: "mult")'
    )
    args = parser.parse_args()
    
    
    #-----------------------------------------------
    
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
        
        
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
    )

    set_seed(args)


    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
    
    label_list=['공포', '놀람', '분노', '슬픔', '중립', '행복', '혐오']
    
    train_data = get_data(args, args.data_name, label_list, "train")
    eval_data = get_data(args, args.data_name, label_list, "dev")
    test_data = get_data(args, args.data_name, label_list, "test")

    bert_args = torch.load(args.bert_args_path)
    

    
    orig_d_a, orig_d_t= args.n_mfcc, args.hidden_size_for_bert
    args.cls_model_name = 'audio_model' if args.do_audio else 'crossModal_model'


    args.d_out = len(label_list)
                                
    model = MULTModel(
        only_audio=args.do_audio,
        merge_how= 'average',
        orig_d_a=orig_d_a,
        orig_d_t=orig_d_t,
        n_head=args.num_heads,
        n_cmlayer=args.layers,
        d_out = args.d_out,
        d_model=args.d_model,
        emb_dropout=args.embed_dropout,
        attn_dropout=args.attn_dropout,
        attn_dropout_audio=args.attn_dropout_a,
        attn_dropout_vision=args.attn_dropout_v,
        relu_dropout=args.relu_dropout,
        res_dropout=args.res_dropout,
        out_dropout=args.out_dropout,
        max_position=128,
        attn_mask=args.attn_mask,
        scale_embedding=True,
    ).to(args.device)


    if args.local_rank == 0:
        torch.distributed.barrier()
    
    # training
    train(args, bert_args, label_list, train_data, eval_data, model)
    
    
    
    # test
    if args.local_rank == -1 or torch.distributed.get_rank() == 0:

        
        
        args = torch.load(f"{args.cls_model_name}/{args.params_name}/args.pt")
        model = torch.load(f"{args.cls_model_name}/{args.params_name}/model.pt").to(args.device)
        
        
        test_collate_fn = AudioTextBatchFunction(args= args,
                                    pad_idx = test_data.pad_idx,
                                    cls_idx = test_data.cls_idx, 
                                    sep_idx = test_data.sep_idx,
                                    bert_args = bert_args,
                                    num_label_from_bert = len(label_list),
                                    device = 'cpu'
                                   )

        tst_loss, tst_acc, tst_macro_f1, (total_y_hat, cm, cr) = evaluate(args, 
                                                      label_list,
                                                      test_data, 
                                                      model, 
                                                      torch.nn.CrossEntropyLoss(), 
                                                      test_collate_fn)
        
        
        # Save results in 'model_saved_finetuning/results.csv'
        test_result_writer = ResultWriter(f"./{args.cls_model_name}/test_results.csv")

        test_results = {
            'tst_loss': tst_loss,
            'tst_acc': tst_acc,
            'tst_macro_f1' : tst_macro_f1
        }
        
        # add f1 score for each class
        test_results.update(cr)

        # summary
        test_result_writer.update(args, **test_results)

        
        
        # confusion matrix
        draw_cm(args,label_list, cm) 
        
        # real/prediction results
        tmp = pd.read_pickle(os.path.join(args.data_path,"test.pkl")) 
        tmp['Pred'] = [label_list[i] for i in total_y_hat]
        tmp[['Emotion','Pred']].to_csv(f'./{args.cls_model_name}/{args.params_name}/test_pred_result.csv')
        
        
if __name__ == "__main__":
    main()
