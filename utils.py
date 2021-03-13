import logging
from os import mkdir, path

import seaborn as sns
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sns.set_style('darkgrid')


def setup_logging(args, run_identifier=''):
    logging.getLogger().setLevel(logging.INFO)
    if args.log_to_file:
        for handler in logging.root.handlers[:]:
            logging.roost.removeHandler(handler)
        log_filename = run_identifier + '.log'
        logging.basicConfig(filename='../logs/' + log_filename,
                            level=logging.INFO,
                            format='%(message)s')
        logging.getLogger().addHandler(logging.StreamHandler())
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s')


def visualize_reconstruction(target, output, filename, loss_func,
                             title='', savefig=False):
    save_dir = '../generated-data/comparisons/{}.jpg'.format(filename)
    target = target.squeeze()
    output = output.detach().squeeze()

    if 'bce' in loss_func:
        output = torch.sigmoid(output)

    line_params = {'alpha': 0.5, 'linewidth': 1}
    viz = plt.figure(figsize=(10, 5))

    plt.plot(target[0, :], 'b', label='target_x', **line_params)
    plt.plot(output[0, :], 'c', label='output_x', **line_params)
    plt.plot(target[1, :], 'r', label='target_y', **line_params)
    plt.plot(output[1, :], 'm', label='output_y',  **line_params)

    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if savefig:
        plt.savefig(save_dir, quality=80)
        plt.close()  # is handled by tensorboard add_figure()?
        return None
    else:
        return viz


def plot_hist(values, title):
    viz = plt.figure(figsize=(6, 3))
    sns.distplot(values)
    # plt.title(title)
    plt.tight_layout()
    # plt.close()
    return viz


def plot_scatter(df, savefig=False):
    dims = list(range(len(df.columns) - 1))
    viz = plt.figure(figsize=(5, 5))
    if len(dims) == 2:
        sns.scatterplot(*dims, data=df, hue='label')
    else:
        ax = viz.add_subplot(111, projection='3d')
        # get 6 most frequent labels
        # TO-DO: Allow more than 6 labels (need to adjust colors)
        labels = df['label'].value_counts()[:6].index
        for label, color in zip(labels, ['r', 'g', 'b', 'c', 'y', 'm']):
            ax.scatter(*df[df['label'] == label][dims].to_numpy().T, c=color, alpha=0.33)

    plt.tight_layout()
    if savefig:
        logging.info('savefig not implemented! Need to pass on filename!')
        plt.close()
        # plt.savefig('./tmp/{}-{}-{}-perplexity{}-iter{}.jpg'.format(
        #     task, method, len(df), 50, 3000), quality=85)
        # plt.close()
    else:
        return viz
