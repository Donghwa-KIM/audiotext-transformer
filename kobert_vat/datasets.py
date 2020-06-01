import torch
from torch.utils.data import Dataset
import pandas as pd
import random
import copy
import re
import html
import unicodedata


class Datasets(Dataset):
    def __init__(self, file_path, label_list=None, pretrained_type='skt', max_len=64):
        self.max_len = max_len
        if label_list is not None:
            multi_label = ['공포', '놀람', '분노', '슬픔', '중립', '행복', '혐오']
            self.label2idx = dict(zip(label_list, range(len(label_list))))
        self.corpus, self.label = self.get_data(file_path)
        self.pretrained_type = pretrained_type
        self.tokenizer, self.vocab = get_pretrained_model(pretrained_type)

        self.pad_idx = self.vocab['[PAD]']
        self.cls_idx = self.vocab['[CLS]']
        self.sep_idx = self.vocab['[SEP]']
        self.mask_idx = self.vocab['[MASK]']

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        # Normalize
        tokens = self.normalize_string(self.corpus[idx])

        # Tokenize by Wordpiece tokenizer
        tokens = self.tokenize(tokens)

        # Change wordpiece to indices
        tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        
        label = self.label[idx] if self.label is not None else None

        return tokens, label

    @staticmethod
    def normalize_string(s):
        s = html.unescape(s)
        s = re.sub(r"[\s]", r" ", s)
        s = re.sub(r"[^a-zA-Z가-힣ㄱ-ㅎ0-9.!?]+", r" ", s)
        return s

    def get_data(self, file_path):
        data = pd.read_pickle(file_path)
        
        # remove duplicates
        data = data[['Sentence','Emotion']].drop_duplicates( ).reset_index(drop=True)
        
        corpus = data['Sentence']
        label = None
        try:
            label = [self.label2idx[l] for l in data['Emotion']]
        except:
            pass
        return corpus, label

    def tokenize(self, tokens):
        if self.pretrained_type == 'etri':
            return self.tokenizer.tokenize(tokens)
        elif self.pretrained_type == 'skt':
            return self.tokenizer(tokens)


def get_pretrained_model(pretrained_type):
    if pretrained_type == 'etri':
        # use etri tokenizer
        from pretrained_model.etri.tokenization import BertTokenizer
        tokenizer_path = './pretrained_model/etri/vocab.korean.rawtext.list'
        tokenizer = BertTokenizer.from_pretrained(
            tokenizer_path, do_lower_case=False)
        vocab = tokenizer.vocab
    elif pretrained_type == 'skt':
        # use gluonnlp tokenizer
        import gluonnlp as nlp
        vocab_path = './pretrained_model/skt/vocab.json'
        tokenizer_path = './pretrained_model/skt/tokenizer.model'
        vocab = nlp.vocab.BERTVocab.from_json(open(vocab_path, 'rt').read())
        tokenizer = nlp.data.BERTSPTokenizer(
            path=tokenizer_path, vocab=vocab, lower=False)
        vocab = tokenizer.vocab.token_to_idx
    else:
        TypeError('Invalid pretrained model type')
    return tokenizer, vocab
