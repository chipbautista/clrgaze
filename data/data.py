import logging

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch import stack
from sklearn.model_selection import train_test_split

from settings import *
from data.utils import pad
from data.transformer import DataTransformer


np.random.seed(RAND_SEED)


class SignalDataset(Dataset):
    """
    Consolidates the different data set so they're processed in the same way.
    Also used as the Dataset class required for PyTorch's DataLoader.

    Specifically handles:
    1. calls method to load and preprocess data for each data set
    2. pads all the signals to the same length
    3. when used as DataLoader dataset. queries the data from the data sets.
    4. when doing SimCLR training, also calls method to transform each signal
    """
    def __init__(self, corpora, args, caller='', **kwargs):
        self.signal_type = args.signal_type
        assert self.signal_type in ['vel', 'pos']
        self.input_column = 'in_{}'.format(self.signal_type)

        self.val_set_size = self.__reformat_val_set(args.val_set_size)
        self.stratify = args.stratify
        if caller == 'trainer':
            assert args.viewing_time > 0
            self.mode = 'unsupervised_contrastive'
            self.transformer = DataTransformer(args)

        else:
            self.mode = 'evaluation'
            self.transformer = None

        # self.normalize = args.rec_loss == 'bce' or args.signal_type == 'pos'
        self.normalize = False
        self.corpora = corpora

        assert args.hz > 0
        self.hz = args.hz
        self.viewing_time = args.viewing_time
        self.num_gaze_points = int(self.hz * self.viewing_time)
        self.train_set, self.val_set = [], []

        for corpus_name, corpus in self.corpora.items():
            corpus.load_data()
            corpus_samples = ['{}|{}'.format(corpus_name, i)
                              for i in range(len(corpus.data))]

            # if kwargs.get('load_to_memory'):
            if True:
                signal = self._get_signal(corpus.data)
                corpus.data[self.input_column] = signal
                corpus.data.drop(['x', 'y'], axis=1, inplace=True)

            train, val = self._train_test_split(corpus_samples, corpus)
            self.train_set.extend(train)
            self.val_set.extend(val)

        if len(self.val_set) > 0:
            self.val_set = SignalDataset_Val(self.val_set,
                                             self.corpora,
                                             self.normalize,
                                             self.signal_type,
                                             self.input_column,
                                             self.mode,
                                             self.transformer)

        logging.info('\nDataset class initialized from {}.'.format(caller))
        logging.info('Hz: {}. View Time (s): {}'.format(
            args.hz, self.viewing_time))
        logging.info('Signal type: {}'.format(self.signal_type))
        logging.info('Normalize: {}'.format(self.normalize))
        logging.info('Training samples: {}'.format(len(self.train_set)))
        logging.info('Validation samples: {}'.format(len(self.val_set)))

    def _get_signal(self, df):
        if self.signal_type == 'vel':
            signal = df['v']

        else:
            if self.normalize:
                signal = df[['x', 'y']].apply(self.normalize_sample, 1)
            else:
                signal = df.apply(lambda r: np.stack(r[['x', 'y']]).T, 1)

        if self.num_gaze_points > 0 and \
                self.mode != 'unsupervised_contrastive':
            signal = signal.apply(lambda x: pad(self.num_gaze_points, x))

        return signal

    def _train_test_split(self, corpus_samples, corpus):
        # to add support for specifying which WHOLE data sets will be
        # used as validation set
        if isinstance(self.val_set_size, list):
            if corpus.name.lower() in self.val_set_size:
                return [], corpus_samples
            else:
                return corpus_samples, []

        if self.val_set_size == 0:
            return corpus_samples, []

        if self.stratify:
            corpus.data['subj_'] = corpus.data.subj.apply(
                lambda x: x.split('-')[-1])
            return train_test_split(
                corpus_samples,
                test_size=self.val_set_size,
                random_state=RAND_SEED,
                stratify=corpus.data['subj_'].to_numpy())
        else:
            return train_test_split(
                corpus_samples,
                test_size=self.val_set_size,
                random_state=RAND_SEED)

    def __reformat_val_set(self, args_val_set):
        try:
            return float(args_val_set)
        except ValueError:
            args_val_set = args_val_set.replace('-', '_')
            return args_val_set.lower().split(',')

    def __getitem__(self, i):
        corpus, idx = self.train_set[i].split('|')
        data = self.corpora[corpus].data

        if type(data) == pd.DataFrame:
            signal = data.iloc[int(idx)][self.input_column]
        else:  # saved time slices
            signal = data[int(idx)]
            if self.normalize and self.signal_type != 'vel':
                signal = self.normalize_sample(signal)

        if self.mode == 'unsupervised_contrastive':
            return stack(self.transformer.transform(signal))
        else:
            return signal.T

    def __len__(self):
        return len(self.train_set)

    def normalize_sample(self, sample):
        if isinstance(sample, np.ndarray):
            sample[:, 0] /= MAX_X_RESOLUTION
            sample[:, 1] /= MAX_Y_RESOLUTION
            return sample

        sample.x /= MAX_X_RESOLUTION
        sample.y /= MAX_Y_RESOLUTION
        return np.array([sample.x, sample.y]).T


class SignalDataset_Val(SignalDataset):
    def __init__(self, samples, corpora, normalize, signal_type, input_column,
                 mode, transformer):
        self.train_set = samples
        self.corpora = corpora
        self.normalize = normalize
        self.signal_type = signal_type
        self.input_column = input_column
        self.mode = mode
        self.transformer = transformer
