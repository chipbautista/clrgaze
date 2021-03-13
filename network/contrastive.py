from torch import nn, manual_seed

from network.encoder import Encoder

from settings import *


manual_seed(RAND_SEED)


class ContrastiveEncoder(nn.Module):
    def __init__(self, args):
        super(ContrastiveEncoder, self).__init__()

        self.representation_layer = args.representation_layer
        self.encoder = Encoder(args, *ENCODER_PARAMS)
        if self.encoder.multiscale:
            self.skip_1 = nn.Linear(ENCODER_FILTERS[0], ENCODER_FILTERS[-1])
            self.skip_2 = nn.Linear(ENCODER_FILTERS[1], ENCODER_FILTERS[-1])

        p_in_dim = self.encoder.out_dim
        p_mid_dim = int(self.encoder.out_dim)

        self.projection = nn.ModuleList([
            nn.Sequential(nn.Linear(p_in_dim, p_mid_dim), nn.ReLU()),
            nn.Linear(p_mid_dim, LATENT_SIZE)
        ])

        # what will be used for downstream
        if self.representation_layer == 0:
            self.latent_size = p_in_dim
        elif self.representation_layer == 1:
            self.latent_size = p_mid_dim
        else:
            assert 1 == 0

    def forward(self, x):
        x = self.encode(x, True)

        for layer in self.projection:
            x = layer(x)

        return x

    def encode(self, x, is_training=False):  # returns representation
        z = self.encoder(x)

        if self.encoder.multiscale:
            z = self.skip_1(z[0]) + self.skip_2(z[1]) + z[2]

        if is_training or self.representation_layer == 0:
            return z
        else:
            for i, layer in enumerate(self.projection):
                z = layer(z)
                if (i + 1) == self.representation_layer:
                    return [z]

