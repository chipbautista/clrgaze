import logging

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


sns.set(style='darkgrid', font_scale=0.8)
EVAL_FIG_DIR = '../generated-data/evals/figures/'


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

    line_params = {'alpha': 0.8, 'linewidth': 1}
    viz = plt.figure(figsize=(7, 5))

    plt.subplot(211)
    plt.plot(target[0, :], 'b', label='target_x', **line_params)
    plt.plot(target[1, :], 'r', label='target_y', **line_params)
    plt.legend()
    plt.subplot(212)
    plt.plot(output[0, :], 'b', label='output_x', **line_params)
    plt.plot(output[1, :], 'r', label='output_y', **line_params)
    plt.legend()

    # Use this if plotting both target and output in one plot
    # plt.plot(target[0, :], 'b', label='target_x', **line_params)
    # plt.plot(output[0, :], 'c', label='output_x', **line_params)
    # plt.plot(target[1, :], 'r', label='target_y', **line_params)
    # plt.plot(output[1, :], 'm', label='output_y',  **line_params)
    # plt.title(title)

    plt.tight_layout()
    if savefig:
        plt.savefig(save_dir, quality=80)
        plt.close()  # is handled by tensorboard add_figure()?
        return None
    else:
        return viz


def visualize_pos_to_vel(pos, hz):
    viz = plt.figure(figsize=(9, 5))
    vel = np.abs(np.diff(pos)) / (1000 / hz)
    line_params = {'alpha': 0.8, 'linewidth': 1}
    plt.subplot(211)
    plt.plot(pos[0, :], 'b', label='pos_x', **line_params)
    plt.plot(pos[1, :], 'r', label='pos_y', **line_params)
    plt.legend()
    plt.subplot(212)
    plt.plot(vel[0, :], 'b', label='vel_x', **line_params)
    plt.plot(vel[1, :], 'r', label='vel_y', **line_params)
    plt.legend()
    plt.tight_layout()
    import pdb; pdb.set_trace()
    return


def visualize_signal(signal):
    viz = plt.figure(figsize=(6, 3))
    plt.plot(signal[0, :], 'b', label='x', linewidth=1)
    plt.plot(signal[1, :], 'r', label='y', linewidth=1)
    return viz


def plot_hist(values, title):
    viz = plt.figure(figsize=(6, 3))
    sns.distplot(values)
    # plt.title(title)
    plt.tight_layout()
    # plt.close()
    return viz


def plot_scatter(df, savefig_title=None, draw_legend=True):
    dims = list(range(len(df.columns) - 1))
    viz = plt.figure(figsize=(5, 5))

    # get 6 most frequent labels
    if len(dims) == 2:
        top_labels = df['label'].value_counts()[:10].index
        draw_legend = False if (not draw_legend or len(top_labels) > 6) else 'full'
        df = df[df.label.isin(top_labels)]

        sns.scatterplot(*dims, data=df, hue='label',
                        palette=sns.color_palette("hls", len(top_labels)),
                        legend=draw_legend)
    else:
        top_labels = df['label'].value_counts()[:6].index
        ax = viz.add_subplot(111, projection='3d')
        # TO-DO: Allow more than 6 labels (need to adjust colors)
        for label, color in zip(top_labels, ['r', 'g', 'b', 'c', 'y', 'm']):
            ax.scatter(*df[df['label'] == label][dims].to_numpy().T, c=color, alpha=0.66)

    plt.tick_params(axis='both', which='both',
                    bottom=False, top=False, left=False, right=False,
                    labelleft=False, labelbottom=False)
    plt.ylabel(None)
    plt.xlabel(None)
    plt.tight_layout()
    if savefig_title:
        f = EVAL_FIG_DIR + savefig_title
        plt.savefig(f, dpi=200, quality=100)
        logging.info('scatterplot saved to {}'.format(f))
        # plt.close()
    return viz


def plot_feature_importance(df, signal_types, classifier='svm_linear', savefig_title=None):
    def plot_counts(_df):
        # to ensure that level 2 is always darker than level 1
        red_seq = sns.color_palette("RdBu", 4)[:2]
        # red_seq = sns.color_palette(['#F6C566', '#FAE3B2'])
        blue_seq = sns.color_palette("RdBu_r", 4)[:2]
        # blue_seq2 = sns.color_palette(['#5B92C5', '#ADC8E2'])
        if signal_types[0] == 'pos':  # red first
            sns.set_palette(red_seq + blue_seq)
        else:
            sns.set_palette(blue_seq + red_seq)
        sns.set_style('whitegrid')

        _df = _df.drop(['index', 'classifier'], 1)
        count_plot = _df.groupby('task').count().plot(kind='bar', stacked=True, rot=0, figsize=(7, 2.5)).legend(loc='upper right', bbox_to_anchor=(1.166, 1.0))

        plt.xlabel(None)
        plt.ylabel('Top 20% feature count')
        plt.tight_layout()
        return count_plot.get_figure()

    def plot_dist(df, signal):
        plt.figure(figsize=(5.5, 3))
        if signal == 'pos':
            sns.set_palette(sns.color_palette("RdBu", 4))
        else:
            sns.set_palette(sns.color_palette("RdBu_r", 4))

        signal_cols = [c for c in df.columns if c.startswith(signal)]
        df = df[['task'] + signal_cols].fillna(0)

        if signal + '-1' in df.columns:  # hierarchical
            # hardcode this so the color/legend is consistent
            c1, c2 = signal + '-1', signal + '-2'

            def rearrange_df(x, c1, c2):
                return (x[c1], c1) if x[c1] > x[c2] else (x[c2], c2)

            df[['fi', 'feature_type']] = df.apply(
                lambda x: pd.Series(rearrange_df(x, c1, c2)), axis=1)
        else:
            df['fi'] = df[signal_cols[0]]
            df['feature_type'] = signal_cols[0]

        # normalize values first per task
        for task in df.task.unique():
            task_df = df[df.task == task]
            df.loc[task_df.index, 'fi'] = task_df['fi'] / task_df['fi'].sum()

        viz = sns.violinplot(
            x='task', y='fi', hue='feature_type', data=df,
            split=True, inner='quart')

        plt.ylabel('Normalized feature importance values')
        plt.tight_layout()

        return viz.get_figure()

    df = df.reset_index()
    _df = df[df.classifier == classifier]

    if len(_df) < 1:
        logging.info('No feature importance values for {} classifier.'.format(
            classifier))
        return

    # need to plot them in separate functions so they don't overlap in 1 plot

    count_plot = plot_counts(_df)
    ret = {'count': count_plot}
    if savefig_title:
        f = EVAL_FIG_DIR + '_'.join([savefig_title, 'fi_count'])
        count_plot.savefig(f, dpi=200, quality=100)
        logging.info('FI Count plot saved to {}'.format(f))

    for s in signal_types:
        dist_plot = plot_dist(_df, s)
        ret[s] = dist_plot
        if savefig_title:
            f = EVAL_FIG_DIR + '_'.join([savefig_title, 'fi_dist', s])
            dist_plot.savefig(f, dpi=200, quality=100)
            logging.info('FI Dist plot saved to {}'.format(f))

    # ret.update({s: plot_dist(_df, s) for s in signal_types})
    # import pdb; pdb.set_trace()
    return ret


# not part of the workflow but only for analysis.
# writing this code here so I can repeat it easily
def tsne_plot_fifa(df):
    from sklearn.manifold import TSNE
    from .classification_tasks import AgeGroupBinary, GenderBinary

    df = df[df.corpus == 'Cerf2007-FIFA']

    z_values = np.stack(df.z)
    # z_values = StandardScaler().fit_transform(np.stack(df.z))

    df_tsne = pd.DataFrame(TSNE(
        2, perplexity=30, learning_rate=500, n_jobs=3).fit_transform(z_values))

    # plot #1: subject
    df_tsne['label'] = df.subj
    fig = plot_scatter(df_tsne, '')
    import pdb; pdb.set_trace()

    # plot #2: gender
    df_tsne['label'] = AgeGroupBinary().get_xy(df)[1]
    fig = plot_scatter(df_tsne, '')

    # plot #3: age group
    df_tsne['label'] = GenderBinary().get_xy(df)[1]
    fig = plot_scatter(df_tsne, '')


def tsne_plot_etra(df):
    from sklearn.manifold import TSNE
    from .classification_tasks import SearchTaskETRA, ETRAStimuli

    df = df[df.corpus == 'ETRA2019']

    z_values = np.stack(df.z)
    # z_values = StandardScaler().fit_transform(np.stack(df.z))

    df_tsne = pd.DataFrame(TSNE(2, perplexity=30, learning_rate=500, n_jobs=3).fit_transform(z_values))

    # plot #1: subject
    df_tsne['label'] = df.subj
    fig = plot_scatter(df_tsne, '')
    import pdb; pdb.set_trace()

    # plot #2: gender
    df_tsne['label'] = SearchTaskETRA().get_xy(df)[1]
    fig = plot_scatter(df_tsne, '')

    # plot #3: age group
    df_tsne['label'] = ETRAStimuli().get_xy(df)[1]
    fig = plot_scatter(df_tsne, '')


def tsne_plot_corpus(df, representation_name):
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    # z_values = np.stack(df.z)
    z_values = StandardScaler().fit_transform(np.stack(df.z))

    df_tsne = pd.DataFrame(TSNE(
        2, perplexity=30, learning_rate=500, n_jobs=3).fit_transform(z_values))

    df_tsne['label'] = list(df.corpus)
    fig = plot_scatter(df_tsne, '', draw_legend=True)

    f = EVAL_FIG_DIR + '_'.join([representation_name, 'tsne_corpus'])
    plt.savefig(f, dpi=200, quality=100)
    logging.info('t-SNE Corpus plot saved to {}'.format(f))
    import pdb; pdb.set_trace()

"""
Code for getting a sample position and velocity signal:
(change variables for pos y, vel x, vel y)

sample = self.data.iloc[50]
plt.figure(figsize=(3,0.5)); plt.plot(sample.x, color='b', alpha=0.75)
plt.tick_params(axis='y', left=False, right=False, labelleft=False)
plt.tight_layout()
plt.savefig('pos_x', dpi=200, quality=100)
"""
