import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import pickle
import os
import soundfile as sf
from tqdm import tqdm
import torchaudio
from collections import defaultdict
import pandas as pd
import re
import html

# best model
from korbert_vat.pretrained_model.etri.tokenization import BertTokenizer

class DataSets(Dataset):
    """ Adapted from original multimodal transformer code"""

    def __init__(self, path, 
                 label_list,
                 data="vat",
                 split_type="train"):
        super(DataSets, self).__init__()
        
        
        self.label2idx = dict(zip(label_list, range(len(label_list))))
        
        self.audio, self.text, self.labels = self.get_data(path)
        self.data_name = data
        self.split_type = split_type
        self.n_modalities = 2  # / text/ audio

        self.tokenizer, self.vocab = self.get_pretrained_model('etri')
        self.pad_idx = self.vocab['[PAD]']
        self.cls_idx = self.vocab['[CLS]']
        self.sep_idx = self.vocab['[SEP]']
        self.mask_idx = self.vocab['[MASK]']
        
    def get_n_modalities(self):
        return self.n_modalities


    def __len__(self):
        return len(self.labels)
    
    def get_data(self, file_path):
        
        data = pd.read_pickle(file_path)
        
        text = data['Sentence']
        audio =  data['audio']
        

        label = [self.label2idx[l] for l in data['Emotion']]

        return audio, text, label
    
    def __getitem__(self, index):
        tokens = self.normalize_string(self.text[index])
        # Tokenize by Wordpiece tokenizer
        tokens = self.tokenize(tokens)
        
        # Change wordpiece to indices
        tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # ------------------------guideline------------------------------------
        # naming as labels -> use to sampler
        # float32 is required for mfcc function in torchaudio
        #----------------------------------------------------------------------
        return self.audio[index].astype(np.float32), tokens, self.labels[index]
    
    @staticmethod
    def normalize_string(s):
        s = html.unescape(s)
        s = re.sub(r"[\s]", r" ", s)
        s = re.sub(r"[^a-zA-Z가-힣ㄱ-ㅎ0-9.!?]+", r" ", s)
        return s
    def tokenize(self, tokens):
        return self.tokenizer.tokenize(tokens)

    
    def get_pretrained_model(self, pretrained_type):
        # use etri tokenizer
        tokenizer_path = './korbert_vat/pretrained_model/etri/vocab.korean.rawtext.list'
        tokenizer = BertTokenizer.from_pretrained(
            tokenizer_path, do_lower_case=False)
        vocab = tokenizer.vocab


        return tokenizer, vocab
    
    
    
