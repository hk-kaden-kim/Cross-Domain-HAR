"""
Create MTSSL pretraining data from the preprocessed capture24 dataset for finetuning.

Inputs
-------------------

Outputs
-------------------
    ├ train
    ⏐   ├ file_list.csv
    ⏐   ├ P*.npy
    ├ test
    ⏐   ├ file_list.csv
    ⏐   ├ P*.npy
"""

import numpy as np
import pandas as pd
import os
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from utils import ftDataset_to_ptDataset

##########################################################################
# GLOBAL VARIABLES!!!

SCRATCH = os.environ['SCRATCH']
FINETUNE_DATASET_DIR = SCRATCH + '/mp-dataset/finetune_capture24_30hz_w10'
OUTDIR = SCRATCH + '/mp-dataset/pretrain_capture24_30hz_w10/data'

xfile = os.path.join(FINETUNE_DATASET_DIR, 'X.npy')
pidfile = os.path.join(FINETUNE_DATASET_DIR, 'pid.npy')

##########################################################################

def main():

    X = np.load(xfile)

    pid = np.load(pidfile)
    pid_unique = np.unique(pid)
    dummy = np.zeros(len(pid_unique))

    pid_unique_train, pid_unique_test, _, _ = train_test_split(pid_unique,dummy,test_size = 0.2,random_state=42,shuffle=True)

    train = os.path.join(OUTDIR, 'train')
    test = os.path.join(OUTDIR, 'test')

    print("Create MTSSL Pretrain dataset(Train) of Capture24...")
    print(pid_unique_train)
    ftDataset_to_ptDataset(train,pid_unique_train,pid,X)
    
    print("Create MTSSL Pretrain dataset(Test) of Capture24...")
    print(pid_unique_test)
    ftDataset_to_ptDataset(test,pid_unique_test,pid,X)

if __name__ == "__main__":
    main()
