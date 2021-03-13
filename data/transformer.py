import logging

import numpy as np
from torch import Tensor
from torch.distributions.normal import Normal

from data.utils import pad


class DataTransformer:
    def __init__(self, args):
        self.slice_limit = int(args.hz * args.viewing_time)
        self.signal_type = args.signal_type
        self.gaussian = Normal(0.0, 0.5)

        self.transforms = [
            self.dropout,
            self.add_gaussian_noise,
            self.dropout_and_noise,
            self.chunk_dropout,
            self.chunk_copy,
            self.chunk_swap,
            # self.point_copy,
            # self.point_swap,
            self.alternate_dropout,
            self.channel_dropout,
            self.identity,
        ]

        self.crop_methods = ['random', 'same', 'consecutive']
        # self.crop_methods = ['consecutive', 'random']

        self.dropout_p = 0.2
        self.chunk_length = 0.2
        logging.info('Data transformer for contrastive learning initialized.')
        logging.info('Dropout p={}'.format(self.dropout_p))
        logging.info('Chunk length={}'.format(self.chunk_length))
        logging.info('Other transformations:\n{}'.format(
            [t.__name__ for t in self.transforms[1:]]))
        logging.info('Crop methods: {}'.format(self.crop_methods))

    def transform(self, x):
        def _fn(x):
            return np.random.choice(self.transforms)(x)

        crop_method = np.random.choice(self.crop_methods)
        if crop_method == 'same':
            crop1 = self.crop(x)
            crop2 = crop1.clone()
        elif crop_method == 'random':
            crop1, crop2 = self.crop(x), self.crop(x)
        else:
            crop1, crop2 = self.consecutive_crop(x)

        return _fn(crop1).T, _fn(crop2).T

    def consecutive_crop(self, x):
        if len(x) > self.slice_limit * 2:
            start_idx = np.random.randint(0, len(x) - self.slice_limit * 2)
            cut_at_idx = start_idx + self.slice_limit
            end_at_idx = cut_at_idx + self.slice_limit
        else:  # just split in half
            start_idx = 0
            cut_at_idx = int(len(x) / 2)
            end_at_idx = None

        crop_1 = x[start_idx: cut_at_idx]
        crop_2 = x[cut_at_idx: end_at_idx]

        if len(crop_1) < self.slice_limit:
            crop_1 = pad(self.slice_limit, crop_1)
        if len(crop_2) < self.slice_limit:
            crop_2 = pad(self.slice_limit, crop_2)

        return Tensor(crop_1), Tensor(crop_2)

    def crop(self, x):
        # Choose which time step to start the crop
        # allow the cropped version to have trailing zeros
        try:
            start = np.random.randint(0, len(x) - self.slice_limit / 2)
            crop = x[start: start + self.slice_limit]
        except ValueError:  # when the original signal's too short to crop
            crop = x

        if len(crop) < self.slice_limit:
            crop = pad(self.slice_limit, crop)

        return Tensor(crop)

    def downsample(self, x):
        x = x[::2]
        x = pad(self.slice_limit, x)
        return Tensor(x)

    def add_gaussian_noise(self, x):
        x = x + self.gaussian.sample(x.shape)
        x[x < 0] = 0
        return x

    def dropout_and_noise(self, x):
        x = self.add_gaussian_noise(x)
        return self.transforms[0](x)

    def chunk_dropout(self, x):
        chunk_size = int(len(x) * self.chunk_length)
        chunk_start = np.random.randint(0, len(x) - chunk_size)
        x[chunk_start: chunk_start + chunk_size] = 0
        return x

    def chunk_copy(self, x):
        chunk_length = int(len(x) * self.chunk_length)
        chunk_start = np.random.randint(chunk_length * 2,
                                        len(x) - chunk_length * 2)
        chunk_end = chunk_start + chunk_length
        chunk = x[chunk_start: chunk_end].clone()

        direction = np.random.choice(['left', 'right'])
        if direction == 'left':
            start = chunk_start - chunk_length
            end = chunk_start
        else:
            start = chunk_end
            end = chunk_end + chunk_length
        x[start: end] = chunk
        return x

    def chunk_swap(self, x):
        # get a random chunk of the sequence
        chunk_length = int(len(x) * self.chunk_length)
        chunk_start = np.random.randint(chunk_length * 2,
                                        len(x) - chunk_length * 2)

        chunk_end = chunk_start + chunk_length
        chunk = x[chunk_start: chunk_end]

        direction = np.random.choice(['left', 'right'])
        if direction == 'left':
            start = chunk_start - chunk_length
            end = chunk_start
        else:
            start = chunk_end
            end = chunk_end + chunk_length

        # swap
        tmp = x[start: end].clone()
        x[start: end] = chunk
        x[chunk_start: chunk_end] = tmp
        return x

    def point_copy(self, x):
        points_to_jitter = np.argwhere(
            np.random.choice([True, False], size=len(x), p=[0.25, 0.75]))
        direction = np.expand_dims(
            np.random.choice([-1, +1], size=len(points_to_jitter)), -1)
        points_to_swap = points_to_jitter + direction

        # avoid IndexError
        within_sample = (points_to_swap >= 0) & (points_to_swap < len(x))
        points_to_jitter = points_to_jitter[within_sample]
        points_to_swap = points_to_swap[within_sample]

        x[points_to_swap] = x[points_to_jitter].clone()

        return x

    def point_swap(self, x):
        points_to_jitter = np.argwhere(
            np.random.choice([True, False], size=len(x), p=[0.2, 0.8]))
        direction = np.expand_dims(
            np.random.choice([-1, +1], size=len(points_to_jitter)), -1)
        points_to_swap = points_to_jitter + direction

        within_sample = (points_to_swap >= 0) & (points_to_swap < len(x))
        points_to_jitter = points_to_jitter[within_sample]
        points_to_swap = points_to_swap[within_sample]

        tmp = x[points_to_swap].clone()
        x[points_to_swap] = x[points_to_jitter]
        x[points_to_jitter] = tmp
        return x

    def dropout(self, x):
        points_to_dropout = np.argwhere(
            np.random.choice([True, False], size=len(x),
                             p=[self.dropout_p, 1 - self.dropout_p]))
        x[points_to_dropout.reshape(-1)] = 0
        return x

    def alternate_dropout(self, x):
        x[::2] = 0
        return x

    def channel_dropout(self, x):
        num_channels = x.shape[1]
        channel = np.random.choice(list(range(num_channels)))
        x[:, channel] = 0
        return x

    def identity(self, x):
        return x
