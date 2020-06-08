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


from train import AudioTextBatchFunction, evaluate, draw_cm

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




def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--text_data", default='test', type=str,
                        help="STT(test_stt) or real Text(test)")

    parser.add_argument("--test_data_path", default='../split_datasets_1channel/test.pkl', type=str,
                        help="test data path")
    parser.add_argument("--best_model_path", default='crossModal_model/best_model/model.pt', type=str,
                        help="test data path")
    parser.add_argument("--args_path", default='crossModal_model/best_model/args.pt', type=str,
                        help="test data path")


    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    args_ = parser.parse_args()

    model = torch.load(args_.best_model_path)
    args = torch.load(args_.args_path)
    
    label_list=['공포', '놀람', '분노', '슬픔', '중립', '행복', '혐오']
    logging.info(label_list)
    test_data = get_data(args, args.data_name, label_list, args_.text_data)
    
    # pretrained bert
    bert_args = torch.load(args.bert_args_path)    
    
    # preprocessing for bert vector and mfcc
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
    print("loss : {} \nacc : {} \nf1 : {}".format(tst_loss, tst_acc, tst_macro_f1))

    # Save results
    args.params_name = 'best_model'

    test_result_writer = ResultWriter(f"./{args.params_name}/test_results.csv")

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
    if args_.text_data=='test':
        tmp = pd.read_pickle(os.path.join(args.data_path,"test.pkl")) 
        tmp['Pred'] = [label_list[i] for i in total_y_hat]
        tmp[['Emotion','Pred']].to_csv(f'./{args.cls_model_name}/{args.params_name}/test_pred_result.csv')

if __name__ == '__main__':
    main()