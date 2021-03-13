# et-representation-learning

This repository contains the training scripts for GazeContrast, split into three folders:
1. `network` - contains network architectures and loss functions
2. `data` - contains scripts to extract, preprocess, and transform data sets
3. `evals` - contains scripts to perform downstream tasks

Main files to run:
1. `train.py` -- does training...
2. `train_supervised.py` -- uses the base encoder network to perform supervised classification
3. `evaluate.py` -- loads pre-trained model and performs downstream tasks
4. `utils.py` -- helper functions
5. `settings.py` -- contains hyperparameters and arg parsing

### Replicating our results
1. Download the data sets and extract the individual folders to `data/`
2. Run `train.py`
    - At the first run, this automatically does extraction and preprocessing of the data sets and saves them into individual `h5` files.
    - Instantiates network architecture and performs contrastive learning

### Citation
If found