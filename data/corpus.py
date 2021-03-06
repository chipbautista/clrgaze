import logging
import time
import pickle
from os import path

import numpy as np
import pandas as pd

import data.utils as du
from settings import *


class EyeTrackingCorpus:
    def __init__(self, args=None):
        if args:
            self.signal_type = args.signal_type
            self.effective_hz = args.hz or self.hz

        self.viewing_time = 0
        self.root = DATA_ROOT + self.root
        self.stim_dir = DATA_ROOT + self.stim_dir if self.stim_dir else None

        self.name = self.__class__.__name__
        self.dir = GENERATED_DATA_ROOT + self.name
        self.data = None

    def load_data(self, load_labels=False):
        self.load_raw_data()

    def load_raw_data(self):
        logging.info('Extracting raw data...'.format(self.name))

        data_file = self.dir + '-data.pickle'
        if not path.exists(data_file):
            extract_start = time.time()
            self.data = pd.DataFrame(
                columns=['subj', 'stim', 'task', 'x', 'y'],
                data=self.extract())
            self.data.x = self.data.x.apply(lambda a: np.array(a))
            self.data.y = self.data.y.apply(lambda a: np.array(a))
            logging.info('- Done. Found {} samples. ({:.2f}s)'.format(
                len(self.data),
                time.time() - extract_start))
            with open(data_file, 'wb') as f:
                pickle.dump(self.data, f)
            logging.info('- Data saved to' + data_file)
        else:
            with open(data_file, 'rb') as f:
                self.data = pickle.load(f)
            logging.info('- Data loaded from' + data_file)

        self.preprocess_data()

    def extract(self):
        """
        Should be implemented by all data sets in corpora.py
        Go through all samples and return a NumPy array of size (N, 4)
        N: number of samples (subjects x stimuli)
        4: columns for the pd DataFrame (subj, stim, x coords, y coords)
        """
        pass

    def append_to_df(self, data):
        self.data = self.data.append(
            dict(zip(['subj', 'stim', 'x', 'y'], list(data))),
            ignore_index=True)

    def __len__(self):
        return len(self.data)

    def preprocess_data(self):
        def preprocess(trial):
            # trim to specified viewing time
            # no trim happens when self.slice_time_windows or contrastive
            trial.x = trial.x[:sample_limit]
            trial.y = trial.y[:sample_limit]

            # always convert blinks (negative values) to 0.
            # trial.x[np.where(trial.x < 0)] = 0
            # trial.y[np.where(trial.y < 0)] = 0

            # # convert negative values and outliers to NaN
            trial.x[np.where(trial.x < 0)] = np.nan
            trial.y[np.where(trial.y < 0)] = np.nan

            # coordinate normalization is not necessary for velocity space
            trial.x = np.clip(trial.x, a_min=0, a_max=self.w or MAX_X_RESOLUTION)
            trial.y = np.clip(trial.y, a_min=0, a_max=self.h or MAX_Y_RESOLUTION)

            trial.x = du.interpolate_nans(trial.x)
            trial.y = du.interpolate_nans(trial.y)

            if self.signal_type == 'pos':
                trial = self.pull_coords_to_zero(trial)

            # scale coordinates so 1 degree of visual angle = 35 pixels
            try:
                scale_value = PX_PER_DVA / self.px_per_dva
                trial[['x', 'y']] *= scale_value
            except AttributeError:  # if corpora has no information about dva
                pass

            if self.resample == 'down':
                trial = du.downsample(trial, self.effective_hz, self.hz)
            elif self.resample == 'up':
                trial = du.upsample(trial, self.effective_hz, self.hz)

            return trial

        sample_limit = (int(self.hz * self.viewing_time)
                        if self.viewing_time > 0
                        else None)

        if (self.hz - self.effective_hz) > 10:
            self.resample = 'down'
        elif (self.effective_hz - self.hz) > 10:
            self.resample = 'up'
        else:
            self.resample = None

        if self.resample:
            logging.info('- Resampling {} to {}.'.format(self.hz, self.effective_hz))

        self.data = self.data.apply(preprocess, 1)

        if 'vel' in self.signal_type:
            logging.info('Calculating velocities...')
            ms_per_sample = 1000 / self.effective_hz
            self.data['v'] = self.data[['x', 'y']].apply(
                lambda x: np.abs(np.diff(np.stack(x))).T, 1) / ms_per_sample
            logging.info(du.get_stats(self.data['v']))

            if self.signal_type == 'posvel':
                self.data['pv'] = self.data.apply(lambda x: np.stack(
                    (x['x'][:-1], x['y'][:-1],
                     x['v'][:, 0], x['v'][:, 1])).T,
                    axis=1)

        else:
            logging.info(du.get_stats(self.data[['x', 'y']]))
