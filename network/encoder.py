import logging
from torch import nn, mul


class Encoder(nn.Module):
    def __init__(self, args, kernel_size, filters, dilations, downsamples,
                 global_pool=True, in_channels=2):
        super(Encoder, self).__init__()
        self.multiscale = args.multiscale
        self.in_dim = int(args.hz * args.viewing_time)
        self.in_channels = in_channels
        self.global_pool = global_pool

        self.kernel_size = kernel_size
        self.filters = filters
        self.dilations = dilations
        self.downsamples = downsamples
        # Do not downsample at the last layer!
        # self.downsamples[-1] = 0

        self.out_dim = self.filters[-1]

        self.blocks = []
        for block_num, (f, dil, down) in enumerate(zip(self.filters,
                                                       self.dilations,
                                                       self.downsamples)):
            residual_block = ResidualBlock(
                in_channels=(self.in_channels if block_num == 0
                             else self.filters[block_num - 1]),
                mid_channels=f,
                out_channels=f,
                dilations=dil,
                kernel_size=self.kernel_size,
                downsample=down,
                squeeze_and_excite=args.squeeze_and_excite
            )
            self.blocks.append(residual_block)

        self.blocks = nn.Sequential(*self.blocks)
        self.out_dim = int(self.out_dim)
        logging.info('\nEncoder initialized.')
        logging.info('Multiscale Rep.: {}'.format(self.multiscale))
        logging.info('Global Pool: {}'.format(self.global_pool))
        logging.info('Downsample amounts per block: {}\n'.format(self.downsamples))

    def forward(self, x):
        if not self.multiscale:
            x = self.blocks(x)
            return self._gap(x)  # Global Average Pooling

        block_features = []
        for block_num, block in enumerate(self.blocks):
            x = block(x)

            block_features.append(self._gap(x))
        return block_features

    def _gap(self, x):
        return x.mean(-1) if self.global_pool else x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, dilations,
                 kernel_size, downsample=0, no_skip=False, **kwargs):
        super(ResidualBlock, self).__init__()
        self.kernel_size = kernel_size

        self.relu = nn.ReLU()
        self.conv1 = self._build_conv_layer(
            in_channels, mid_channels, dilations[0])
        self.bn1 = nn.BatchNorm1d(mid_channels)

        self.conv2 = self._build_conv_layer(
            mid_channels, out_channels, dilations[1])
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.skip_conv = (nn.Conv1d(in_channels, out_channels, 1)
                          if not no_skip else None)

        if downsample > 0:
            self.downsample = nn.MaxPool1d(downsample, downsample)
        else:
            self.downsample = None

        if kwargs.get('squeeze_and_excite'):
            self.se_block = SqueezeAndExciteBlock(mid_channels, r=4)

    def _build_conv_layer(self, in_ch, out_ch, dilation):
        if dilation >= 1:
            padding = dilation * (self.kernel_size - 1)
            pad_sides = int(padding / 2)
            return nn.Sequential(
                nn.ConstantPad1d(pad_sides, 0),
                nn.Conv1d(in_ch, out_ch, self.kernel_size, dilation=dilation))
        else:
            return nn.Conv1d(in_ch, out_ch, self.kernel_size)

    def forward(self, x):
        out = self.bn1(self.relu(self.conv1(x)))

        out = self.conv2(out)

        try:
            out = self.se_block(out)
        except AttributeError:
            pass

        if self.skip_conv is not None:
            out = out + self.skip_conv(x)
        out = self.bn2(self.relu(out))

        if self.downsample is not None:
            out = self.downsample(out)
        return out


class SqueezeAndExciteBlock(nn.Module):
    def __init__(self, in_dim, r):
        super(SqueezeAndExciteBlock, self).__init__()
        squeeze_dim = int(in_dim / r)

        self.se = nn.Sequential(
            nn.Linear(in_dim, squeeze_dim),
            nn.ReLU(),
            nn.Linear(squeeze_dim, in_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return mul(self.se(x.mean(-1)).unsqueeze(-1), x)
