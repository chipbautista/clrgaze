import logging
from datetime import datetime
from math import ceil

import pandas as pd
import numpy as np
from torch import no_grad, Tensor, manual_seed
from torch.utils.tensorboard import SummaryWriter
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, GridSearchCV, KFold
from sklearn.dummy import DummyClassifier

from data import get_corpora
from data.data import SignalDataset
from network import ModelManager
from evals.classification_tasks import *
from evals.classifier_settings import *
from evals.utils import *
from settings import *


np.random.seed(RAND_SEED)
manual_seed(RAND_SEED)


class RepresentationEvaluator:
    def __init__(self, tasks, classifiers='all', args=None, **kwargs):
        logging.info('\n---------- Initializing evaluator ----------')

        self.save_tsne_plot = args.save_tsne_plot
        self.scatterplot_dims = 2
        self.viewing_time = kwargs.get('viewing_time') or args.viewing_time
        self.batch_size = args.batch_size if args.batch_size < 256 else 256
        self.tasks = tasks
        self.scorers = ['accuracy']
        if classifiers == 'all':
            self.classifiers = list(CLASSIFIER_PARAMS.values())
        else:
            self.classifiers = [CLASSIFIER_PARAMS[c] for c in classifiers]

        # evaluate while training; use model passed in by trainer
        if 'model' in kwargs:
            self._caller = 'trainer'
            self.model = kwargs['model']
            self.representation_name = kwargs['representation_name']
            self.signal_types = [args.signal_type]
            self.feature_type_idxs = []
            self.dataset = {args.signal_type: kwargs['dataset']}
            # initialize own data set if the trainer is using sliced samples
            if not kwargs['dataset']:
                self.dataset = self.init_dataset(args, **kwargs)

        # evaluate using pretrained models
        elif args.model_pos or args.model_vel:
            self._caller = 'main'
            self.model = ModelManager(args, training=False)
            self.representation_name = '{}-pos-{}-vel-{}'.format(
                run_identifier,
                args.model_pos.split('/')[-1],
                args.model_vel.split('/')[-1])
            self.signal_types = list(self.model.network.keys())
            self.feature_type_idxs = self._build_representation_index_ranges()
            self.dataset = self.init_dataset(args)

        # will store each corpus' info into a unified self.df.
        # this will hold each trial's representation
        self.df = pd.DataFrame(columns=['corpus', 'subj', 'stim', 'task'])

        self.tensorboard = (SummaryWriter(
            'tensorboard_evals/{}'.format(self.representation_name))
            if args.tensorboard else None)

        self.consolidate_corpora()

    def consolidate_corpora(self):
        """
        Constructs own data frame self.df that holds holds input data and
        metadata for each sample
        """
        def get_corpus_df(corpus_name, signal_type):
            corpus = self.dataset[signal_type].corpora[corpus_name]

            corpus.data['corpus'] = corpus_name
            return corpus.data

        # _signal = self.signal_types[0]  # <-- for when I was doing pos + vel
        _signal = 'vel'
        if self.dataset[_signal].stratify:
            # perform classification tasks only on the validation set
            # Not really optimal, I could collate the whole validation set
            # with matching metadata within data.py...
            df_rows = []
            for sample in self.dataset['vel'].val_set.train_set:
                corpus, idx = sample.split('|')
                row = self.dataset['vel'].corpora[corpus].data.iloc[int(idx)].copy()
                row['corpus'] = corpus
                df_rows.append(row)
            self.df = pd.DataFrame(df_rows)
        else:
            for corpus_name in self.dataset[_signal].corpora.keys():
                corpus_df = get_corpus_df(corpus_name, self.signal_types[0])
                logging.info('{} {} signals loaded. Found {} trials'.format(
                    corpus_name, self.signal_types[0], len(corpus_df)))

                self.df = pd.concat([self.df, corpus_df], sort=False)

        logging.info('Loaded corpora ({}) to Evaluator. {} total trials found'.format(
            self.signal_types, len(self.df)))

    def init_dataset(self, args, **kwargs):
        datasets = {}
        for signal_type in self.signal_types:
            args.signal_type = signal_type

            # hacky. tmp :(
            if 'contrastive' in self.representation_name:
                args.loss_type = None
                args.viewing_time = kwargs.get('viewing_time') or args.viewing_time
                logging.info('[evaluator] modified loss type -> {}, viewing time -> {}'.format(
                    args.loss_type, args.viewing_time))

            corpora = get_corpora(args)
            datasets[signal_type] = SignalDataset(corpora, args,
                                                  # caller='evaluator',
                                                  load_to_memory=True,
                                                  **kwargs)
        return datasets

    def extract_representations(self, e=None, log_stats=False):
        z_cols = []
        for signal_type in self.signal_types:
            logging.info('\nExtracting {} representations...'.format(
                signal_type))
            # dataset = self.dataset[signal_type]
            z_col = 'z_' + signal_type
            z_cols.append(z_col)
            in_col = 'in_' + signal_type

            try:
                network = self.model.network[signal_type]
            except TypeError:
                network = self.model.network

            _input = (np.stack(self.df[in_col]) if self.viewing_time > 0
                      else self.df[in_col])

            self.df[z_col] = self._extract_representations(network, _input)

        self.df['z'] = self.df.apply(
            lambda x: np.concatenate([x[col] for col in z_cols]),
            axis=1)

        logging.info('Done. Final representation shape: {}'.format(
            self.df['z'].iloc[0].shape))

        # for analysis only! I run these manually
        if self._caller != 'trainer':
            # tsne_plot_fifa(self.df)
            # tsne_plot_etra(self.df)
            tsne_plot_corpus(self.df, self.representation_name)

    def _extract_representations(self, network, x):
        if len(x.shape) > 2:
            batch_size = self.batch_size
            if x.shape[1] > x.shape[2]:
                x = x.swapaxes(1, 2)
        else:
            x = x.to_numpy()
            batch_size = 1

        reps = []
        network.eval()
        with no_grad():
            for s in range(ceil(len(x) / batch_size)):
                if batch_size == 1:
                    batch = Tensor(x[s]).T.unsqueeze(0)
                else:
                    batch = Tensor(x[batch_size * s: batch_size * (s + 1)])

                reps.append(network.encode(batch.cuda()
                                           )[0].cpu().detach().numpy())
        return reps

    def evaluate(self, e=None):
        scores = {}
        for i, task in enumerate(self.tasks):
            _task = task.__class__.__name__  # convenience var
            logging.info('\nTask {}: {}'.format(i + 1, _task))
            x, y, = task.get_xy(self.df)
            if len(x) < 1:
                continue

            n_fold, refit, test_set = 5, False, None
            if _task == 'Biometrics_EMVIC':  # to compare with LPiTrack
                n_fold, refit, test_set  = 4, 'accuracy', task.get_test(self.df)
            if _task == 'ETRAStimuli_NonBlank':
                n_fold = kf = KFold(len(x))
                logging.info('Performing LOOCV.')
                logging.info(kf)

            self._log_labels(x, y)
            self._write_scatterplot(_task, x, y, e)
            # self._run_dummy_classifier(x, y)

            scores[task.__class__.__name__] = {}
            for classifier, params_grid in self.classifiers:
                if self._caller != 'trainer' and classifier[0] == 'svm_linear':
                    refit = 'accuracy'  # for feature importances

                pipeline = Pipeline([('scaler', StandardScaler()), classifier])

                grid_cv = GridSearchCV(pipeline, params_grid, cv=n_fold,
                                       n_jobs=-1,
                                       scoring=self.scorers,
                                       refit=refit)
                grid_cv.fit(np.stack(x), y)

                # f1 = grid_cv.cv_results_['mean_test_f1_micro'].max()
                acc = grid_cv.cv_results_['mean_test_accuracy'].max()
                logging.info('[{}] Acc: {:.4f}'.format(classifier[0], acc))

                scores[_task][classifier[0]] = acc

                if test_set is not None:
                    x_, y_ = test_set
                    # self._log_labels(x_, y_)
                    test_acc = grid_cv.score(np.stack(x_), y_)
                    logging.info('Test Acc: {:.4f}'.format(test_acc))
                    scores[_task + '_test'] = {}
                    scores[_task + '_test'][classifier[0]] = test_acc
                    self._write_scatterplot(_task, x_, y_, e)

        return scores

    def _log_labels(self, x, y):
        if True:
        # if self._caller == 'main':
            labels = np.unique(y, return_counts=True)
            logging.info(
                '{} samples, {} Classes: '.format(len(x), len(labels[0])))
            logging.info(
                'Class Counts: {}'.format(dict(zip(*map(list, labels)))))

    def _write_scatterplot(self, task, x, y, e):
        def add_figure(df, method, title_suffix=''):
            title = '_'.join([task, method])

            if e is not None:  # means training, dont save each plot to disk
                savefig_title = None
            else:
                savefig_title = self.representation_name + '_' + title + title_suffix
            fig = plot_scatter(df, savefig_title)

            if not self.tensorboard:
                return
            self.tensorboard.add_figure(title, fig, global_step=e)
            logging.info('{} scatterplot saved to tensorboard'.format(method))

        if self.save_tsne_plot:
            z_values = StandardScaler().fit_transform(np.stack(x))
            df = pd.DataFrame(TSNE(self.scatterplot_dims,
                                   perplexity=30,
                                   learning_rate=500,
                                   n_jobs=3).fit_transform(z_values))
            df['label'] = list(y)
            add_figure(df, 'tSNE', '-p{}-lr{}'.format(30, 500))

    def _run_dummy_classifier(self, x, y):
        dummy_pipeline = Pipeline([
            ('scaler', StandardScaler()), ('dummy', DummyClassifier())])
        dummy_scores = cross_validate(
            dummy_pipeline, np.stack(x), y, cv=5, scoring=self.scorers)
        logging.info('Chance Mean Acc: {:.2f}'.format(
            np.mean(dummy_scores['test_accuracy'])))


if __name__ == '__main__':
    run_identifier = 'eval_' + datetime.now().strftime('%m%d-%H%M')
    args = get_parser().parse_args()
    setup_logging(args, run_identifier)
    evaluator = RepresentationEvaluator(tasks=[
        Biometrics_EMVIC(),
        Biometrics_3(),
        Biometrics_All(),
        Biometrics_FIFA(),
        Biometrics_ETRA(),
        Biometrics_MIT(),
        Biometrics_MIT_LTP(),
        Biometrics_MIT_LR(),
        Biometrics_MIT_CVCL(),
    ], args=args)
    evaluator.extract_representations()
    evaluator.evaluate()
