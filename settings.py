import logging
from argparse import ArgumentParser

# training settings
MAX_TRAIN_ITERS = 36000
CHECKPOINT_INTERVAL = 500

# network settings
LATENT_SIZE = 32
ENCODER_FILTERS = [128, 256, 512]
ENCODER_DILATIONS = [(1, 1), (1, 2), (4, 8)]
ENCODER_DOWNSAMPLE = [2, 2, 2]
ENCODER_KERNEL_SIZE = 3

ENCODER_PARAMS = [ENCODER_KERNEL_SIZE, ENCODER_FILTERS, ENCODER_DILATIONS,
                  ENCODER_DOWNSAMPLE]


# data settings
MAX_X_RESOLUTION = 1280
MAX_Y_RESOLUTION = 1024
DATA_ROOT = '../data/'
GENERATED_DATA_ROOT = '../generated-data/'

PX_PER_DVA = 35  # pixels per degree of visual angle

RAND_SEED = 123


def print_settings():
    logging.info({
        k: v for (k, v) in globals().items()
        if not k.startswith('_') and k.isupper()})


def get_parser():
    parser = ArgumentParser()
    parser.add_argument("-l", "--log-to-file", default=False,
                        action="store_true")
    parser.add_argument("-v", "--verbose", default=False,
                        action="store_true")
    parser.add_argument("--save-model", default=False, action="store_true")
    parser.add_argument("--tensorboard", default=False, action="store_true")
    parser.add_argument("--tau", default=0.3, type=float)
    # Encoder Settings
    parser.add_argument("--multiscale", default=False, action="store_true")
    parser.add_argument("--squeeze-and-excite", default=False, action="store_true")
    # Data Settings
    parser.add_argument("-hz", default=0, type=int)
    parser.add_argument("-vt", "--viewing-time",
                        help="Cut raw gaze samples to this value (seconds)",
                        default=-1, type=float)
    parser.add_argument("--signal-type", default='vel', type=str,
                        help="'pos' or 'vel'")
    # Training Settings
    parser.add_argument("--val-set-size", default=0)
    parser.add_argument("--stratify", default=False, action="store_true")
    parser.add_argument("--cuda", default=True, type=bool)
    parser.add_argument("--use-fp16", default=False, action="store_true")
    parser.add_argument("-bs", "--batch-size", default=64, type=int)
    parser.add_argument("-e", "--epochs", default=200, type=int)
    parser.add_argument("-lr", "--learning-rate", default=5e-4, type=float)
    parser.add_argument("--model-pos", default='', type=str)
    parser.add_argument("--model-vel", default='', type=str)
    parser.add_argument("--name-prefix", default='', type=str)
    parser.add_argument("--eval-checkpoint", default=10, type=int)
    # Evaluation Settings
    parser.add_argument("--save-tsne-plot", default=False, action="store_true")
    parser.add_argument("-cv", "--cv-folds", default=5, type=int)
    parser.add_argument("--representation-layer", default=0, type=int,
                        help="0: TCN output, 1: 1st projection layer")
    return parser
