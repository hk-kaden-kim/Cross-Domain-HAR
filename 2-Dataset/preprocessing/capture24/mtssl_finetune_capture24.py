"""
Create MTSSL finetuning data from the capture24 dataset.

Inputs
-------------------

Outputs
-------------------
    X.npy
    y.npy
    pid.npy
    time.npy
"""

import re
import glob
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from utils import resize, is_good_quality

import os

##########################################################################
# GLOBAL VARIABLES!!!

SCRATCH = os.environ['SCRATCH']

DATADIR = SCRATCH + '/mp-dataset/capture24/capture24'

DATAFILES = os.path.join(DATADIR, 'P*.csv.gz')
ANNOLABELFILE = os.path.join(DATADIR, 'annotation-label-dictionary.csv')

DEVICE_HZ = 100  # Hz
LABEL = 'label:Willetts2018' #'label:Walmsley2020' 'label:WillettsSpecific2018' 'label:WillettsMET2018' 'label:DohertySpecific2018' 'label:Willetts2018' 'label:Doherty2018'

WINDOW_SEC = 10  # seconds
WINDOW_OVERLAP_SEC = 0  # seconds
WINDOW_LEN = int(DEVICE_HZ * WINDOW_SEC)  # device ticks
WINDOW_OVERLAP_LEN = int(DEVICE_HZ * WINDOW_OVERLAP_SEC)  # device ticks
WINDOW_STEP_LEN = WINDOW_LEN - WINDOW_OVERLAP_LEN  # device ticks
WINDOW_TOL = 0.01  # 1%

TARGET_HZ = 30  # Hz
TARGET_WINDOW_LEN = int(TARGET_HZ * WINDOW_SEC)
OUTDIR = SCRATCH + f"/mp-dataset/finetune_capture24_{TARGET_HZ}hz_w{WINDOW_SEC}"
OUTDIR = os.path.join(OUTDIR, LABEL[len('label:'):])
##########################################################################


def main():
    # Read annotation label table
    annolabel = pd.read_csv(ANNOLABELFILE, index_col='annotation')

    X, Y, T, P, = [], [], [], []
    print("Create MTSSL Finetune dataset of Capture24...")
    print(LABEL)
    for datafile in tqdm(glob.glob(DATAFILES)):
        data = pd.read_csv(datafile, parse_dates=['time'], index_col='time',
                        dtype={'x': 'f4', 'y': 'f4', 'z': 'f4', 'annotation': 'str'})

        p = re.search(r'(P\d{3})', datafile, flags=re.IGNORECASE).group()

        for i in range(0, len(data), WINDOW_STEP_LEN):

            # Split dataframe with the size of window
            w = data.iloc[i:i + WINDOW_LEN]

            # Check the quality of windows
            if not is_good_quality(w, WINDOW_LEN, WINDOW_SEC, WINDOW_TOL):
                continue
            
            # Extract values from the windows
            t = w.index[0].to_datetime64()
            x = w[['x', 'y', 'z']].values
            y = annolabel.loc[w['annotation'][0], LABEL]
            
            X.append(x)
            Y.append(y)
            T.append(t)
            P.append(p)

    X = np.asarray(X)
    Y = np.asarray(Y)
    T = np.asarray(T)
    P = np.asarray(P)

    # downsample to 30 Hz
    X = resize(X, TARGET_WINDOW_LEN)

    os.system(f'mkdir -p {OUTDIR}')
    np.save(os.path.join(OUTDIR, 'X'), X)
    np.save(os.path.join(OUTDIR, 'Y'), Y)
    np.save(os.path.join(OUTDIR, 'time'), T)
    np.save(os.path.join(OUTDIR, 'pid'), P)

    print(f"Saved in {OUTDIR}")
    print("X shape:", X.shape)
    print("Y distribution:")
    print(pd.Series(Y).value_counts())
    print("User distribution:", len(set(P)))
    print(pd.Series(P).value_counts())

if __name__ == "__main__":
    main()

