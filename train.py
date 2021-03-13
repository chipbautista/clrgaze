"""
python train.py --signal-type=vel -bs=128 -vt=2 -hz=500 --multiscale --slice-time-windows=2s-overlap --autoregressive=false --save-model --save-tsne-plot -l --tensorboard

nohup python -u train.py -hz=500 --eval-checkpoint=200 --name-prefix=amihan_6ds_new_tasks_ --signal-type=vel --squeeze-and-excite --tensorboard --use-fp16 -vt=1 -bs=512 --save-model > nohup.out
"""

import time
import logging
from datetime import datetime

import numpy as np
from torch import manual_seed, no_grad, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda import amp

from data import get_corpora
from data.data import SignalDataset
from data.transformer import DataTransformer
from network import ModelManager
from network.losses import NT_XentLoss
from evaluate import *
from evals.classification_tasks import *
from evals.utils import *
from settings import *

np.random.seed(RAND_SEED)
manual_seed(RAND_SEED)


class Trainer:
    def __init__(self):
        self.model = ModelManager(args)
        self.eval_checkpoint = args.eval_checkpoint
        self.save_model = args.save_model
        self.cuda = args.cuda
        self.use_fp16 = args.use_fp16
        if self.cuda:
            self.model.network = self.model.network.cuda()

        # self.lr_decay = optim.lr_scheduler.StepLR(
        #     self.model.optim, step_size=100, gamma=0.5)

        self._load_data()
        self.loss_fn = NT_XentLoss(args)
        if self.use_fp16:
            self.grad_scaler = amp.GradScaler()
        self.data_transformer = DataTransformer(args)
        self.global_losses = {'train': [], 'val': []}
        self._init_evaluator(args.name_prefix)

    def _load_data(self):
        self.dataset = SignalDataset(get_corpora(args), args,
                                     caller='trainer')

        _loader_params = {'batch_size': args.batch_size, 'shuffle': True,
                          'pin_memory': True, 'num_workers': 4,
                          'drop_last': True}

        self.dataloader = DataLoader(self.dataset, **_loader_params)
        _loader_params.update({'drop_last': False})
        self.val_dataloader = (
            DataLoader(self.dataset.val_set, **_loader_params)
            if self.dataset.val_set else None)

    def _init_evaluator(self, prefix):
        # for logging out this run
        _rep_name = '{}{}{}-hz:{}-s:{}-tau:{}-bs:{}'.format(
            prefix, run_identifier, 'contrastive',
            self.dataset.hz, self.dataset.signal_type,
            self.loss_fn.tau, self.dataloader.batch_size)

        self.evaluator = RepresentationEvaluator(
            tasks=[# AgeGroupBinary(),
                   # GenderBinary(),
                   SearchTaskETRA(),
                   SearchTaskAll(),
                   # ETRAStimuli(),
                   # ETRAStimuli_Fixation(),
                   Biometrics_EMVIC(),
                   Biometrics_ETRA(),
                   Biometrics_ETRA_Fixation(),
                   Biometrics_ETRA_All(),
                   Biometrics_FIFA(),
                   Biometrics_MIT_LTP(),
                   Biometrics_MIT_LR(),
                   Biometrics_MIT_CVCL()],
            classifiers=['svm_linear'],
            args=args,
            model=self.model,
            # dataset=None,  # make evaluator initialize its own -- why?
            dataset=self.dataset,
            representation_name=_rep_name,
            # to evaluate on whole viewing time
            viewing_time=-1)

        if args.tensorboard:
            self.tensorboard = SummaryWriter(
                'tensorboard_runs/{}'.format(_rep_name))
        else:
            self.tensorboard = None

    def _reset_epoch_losses(self):
        self.epoch_losses = {'train': 0.0, 'val': 0.0}

    def _update_global_losses(self):
        for dset in ['train', 'val']:
            self.global_losses[dset].append(self.epoch_losses[dset])

    def train(self):
        logging.info('\n===== STARTING TRAINING =====')
        logging.info('{} samples, {} batches.'.format(
                     len(self.dataset), len(self.dataloader)))
        logging.info('Loss Fn:' + str(self.loss_fn))

        i, e = 0, 0
        _checkpoint_start = time.time()
        while i < MAX_TRAIN_ITERS:
            self._reset_epoch_losses()

            # TRAINING SET
            for b, batch in enumerate(self.dataloader):
                self.model.network.train()
                self.forward(batch, is_training=True)
                i += 1

            # VALIDATION SET
            if self.val_dataloader is not None:
                self.model.network.eval()
                with no_grad():
                    for val_batch in self.val_dataloader:
                        self.forward(val_batch, is_training=False)

            self._update_global_losses()
            self.log(i, time.time() - _checkpoint_start)

            e += 1
            # self.lr_decay.step()
            if e % self.eval_checkpoint == 0:
                self.evaluate_representation(i)

            self.step_operations(e)
            _checkpoint_start = time.time()

        self.evaluate_representation(i)

    def forward(self, batch, is_training=False):
        def _forward(b):
            out = self.model.network(b)
            loss = self.loss_fn(out)
            return out, loss

        batch = batch.float()
        batch = batch.reshape(-1, batch.shape[2], self.dataset.num_gaze_points)

        if self.cuda:
            batch = batch.cuda()
        if self.use_fp16:
            with amp.autocast():
                out, loss = _forward(batch)
        else:
            out, loss = _forward(batch)

        dset = 'train' if is_training else 'val'
        self.epoch_losses[dset] += loss.item()

        if self.model.network.training:
            if self.use_fp16:
                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.step(self.model.optim)
                self.grad_scaler.update()
            else:
                loss.backward()
                self.model.optim.step()
            self.model.optim.zero_grad()

        return

    def evaluate_representation(self, i):
        self.evaluator.extract_representations(i)
        scores = self.evaluator.evaluate(i)
        if self.tensorboard:
            for task, classifiers in scores.items():
                for classifier, acc in classifiers.items():
                    self.tensorboard.add_scalar(
                        '{}_{}_acc'.format(task, classifier), acc, i)
        if self.save_model:
            self.model.save(i, run_identifier, self.global_losses)

    def step_operations(self, e):
        # if e % 1 == 0:
        #     self.model.network.decrement_teacher_forcing_p()
        # self.loss_fn.beta.step()
        # self.lr_decay.step()
        return

    def log(self, i, t):
        mean_train_loss = self.epoch_losses['train'] / len(self.dataloader)
        if self.val_dataloader is None:
            mean_val_loss = 0
        else:
            mean_val_loss = self.epoch_losses['val'] / len(self.val_dataloader)

        logging.info('[{}/{}] TLoss: {:.4f}, VLoss: {:.4f} ({:.2f}s)'.format(
            i, MAX_TRAIN_ITERS, mean_train_loss, mean_val_loss, t))

        if self.tensorboard:
            self.tensorboard.add_scalar('train_total_loss', mean_train_loss, i)
            if mean_val_loss > 0:
                self.tensorboard.add_scalar('val_total_loss', mean_val_loss, i)


args = get_parser().parse_args()
run_identifier = datetime.now().strftime('%m%d-%H%M')
setup_logging(args, run_identifier)
print_settings()

logging.info('\nRUN: ' + run_identifier + '\n')
logging.info(str(args))

trainer = Trainer()
trainer.train()
