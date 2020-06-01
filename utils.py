import torch
import os
import logging
from datasets import DataSets
from datetime import datetime
import pandas as pd
logger = logging.getLogger(__name__)


def get_data(args, dataset, label_list, split="train"):
    """ Adapted from original multimodal transformer code """
    data_path = os.path.join(args.data_path, f"{split}.pkl")
    
    return DataSets(data_path, label_list, dataset, split)




class TqdmLoggingHandler(logging.StreamHandler):
    """ logging handler for tqdm """

    def __init__(self, level=logging.NOTSET):
        super(TqdmLoggingHandler, self).__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


class ResultWriter:
    def __init__(self, dir):
        """ Save training Summary to .csv 
        input
            args: training args
            results: training results (dict)
                - results should contain a key name 'val_loss'
        """
        self.dir = dir
        self.hparams = None
        self.load()
        self.writer = dict()

    def update(self, args, **results):
        now = datetime.now()
        date = '%s-%s-%s %s:%s' % (now.year, now.month, now.day, now.hour, now.minute)
        self.writer.update({'date': date})
        self.writer.update(results)
        self.writer.update(vars(args))

        if self.hparams is None:
            self.hparams = pd.DataFrame(self.writer, index=[0])
        else:
            self.hparams = self.hparams.append(self.writer, ignore_index=True)
        self.save()

    def save(self):
        assert self.hparams is not None
        self.hparams.to_csv(self.dir, index=False)

    def load(self):
        path = os.path.split(self.dir)[0]
        if not os.path.exists(path):
            os.makedirs(path)
            self.hparams = None
        elif os.path.exists(self.dir):
            self.hparams = pd.read_csv(self.dir)
        else:
            self.hparams = None
            