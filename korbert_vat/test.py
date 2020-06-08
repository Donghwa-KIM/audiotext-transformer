from pytorch_transformers.modeling_bert import BertConfig
from pytorch_transformers import optimization

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import Datasets
from model import BertForEmotionClassification
from train_classification import ClassificationBatchFunction, evaluate
from optim import layerwise_decay_optimizer

from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import time
import pickle
import utils
import logging
import argparse
import random
import os
import warnings
import itertools

logger = utils.get_logger('BERT Classification')
logger.setLevel(logging.INFO)

# pretrained_model_path = './senti_model/best_model.bin'
# config_path = './senti_model/bert_config.json'
# args_path = './senti_model/training_args.bin'

pretrained_model_path = './best_model/best_model.bin'
config_path = './best_model/bert_config.json'
args_path = './best_model/training_args.bin'


device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
label_list = ['공포', '놀람', '분노', '슬픔', '중립', '행복', '혐오']

def test(model, args):
    model.eval()

    # Load Datasets
    dataset = Datasets(file_path=args.test_data_path,
                       label_list=label_list,
                       pretrained_type=args.pretrained_type)
    # Use custom batch function
    collate_fn = ClassificationBatchFunction(
        args.max_len, dataset.pad_idx, dataset.cls_idx, dataset.sep_idx)
    loader = DataLoader(dataset=dataset,
                        batch_size=args.train_batch_size,
                        num_workers=8,
                        pin_memory=True,
                        collate_fn=collate_fn)

    loss, acc, f1, (total_y_hat, cm) = evaluate(args, loader, model, device)
    return loss, acc, f1, total_y_hat, cm

def draw_cm(cm):
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
    plt.xlim(-1, len(label_list)-0.5)
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
    if not os.path.exists('result'):
        os.mkdir('result')
    plt.savefig('result/test_result.png', dpi=300)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_path", default='./data/test_stt.pkl', type=str,
                        help="test data path")
    args_ = parser.parse_args()
    pretrained = torch.load(pretrained_model_path)
    args = torch.load(args_path)
    args.test_data_path = args_.test_data_path
    args.eval_batch_size = 64

    bert_config = BertConfig(config_path)
    bert_config.num_labels = 7

    model = BertForEmotionClassification(bert_config).to(device)
    model.load_state_dict(pretrained, strict=False)
    args.n_gpu = 2
    loss, acc, f1, total_y_hat, cm = test(model, args)
    print("loss : {} \nacc : {} \nf1 : {}".format(loss, acc, f1))
    
    draw_cm(cm) 
    tmp = pd.read_pickle(args.test_data_path)
        
    # remove duplicates
    tmp = tmp[['Sentence','Emotion']].drop_duplicates( ).reset_index(drop=True)
        
    tmp['Pred'] = [label_list[i] for i in total_y_hat]
    tmp.to_csv('./result/test_result.csv')
    print("results are saved to result folder")
    
if __name__ == '__main__':
    main()