"""
Create MTSSL finetuning data from the newcastle dataset.

Inputs
-------------------

Outputs
-------------------
    X.npy
    y.npy
    pid.npy
"""

import numpy as np
from tqdm import tqdm
import pandas as pd
import os
import glob
from utils import resize, is_good_quality, label_mapto_capture24, process_files, labeling, label_sleep, extract_number
from scipy.spatial.distance import cdist
import argparse
import re

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--relabel", default=False, action='store_true', help='relabel from original labels to capture24 labels')
    args=parser.parse_args()
    return args

##########################################################################
# GLOBAL VARIABLES!!!

SCRATCH = os.environ['SCRATCH']

FOLDER = f"{SCRATCH}/mp-dataset/newcastlesleep/dataset_psgnewcastle2015_v1.0/acc/"
INITIAL_DATAFILES = glob.glob(os.path.join(FOLDER, "*.csv"))
DATAFILES = glob.glob(f"{SCRATCH}/mp-dataset/newcastlesleep/dataset_psgnewcastle2015_v1.0/acc/merged/*.npy")


DEVICE_HZ = 85.7  # Hz
WINDOW_SEC = 10  # seconds
WINDOW_OVERLAP_SEC = 5  # seconds
WINDOW_LEN = int(DEVICE_HZ * WINDOW_SEC)  # device ticks
WINDOW_OVERLAP_LEN = int(DEVICE_HZ * WINDOW_OVERLAP_SEC)  # device ticks
WINDOW_STEP_LEN = WINDOW_LEN - WINDOW_OVERLAP_LEN  # device ticks
WINDOW_TOL = 0.01  # 1%
TARGET_HZ = 30  # Hz
TARGET_WINDOW_LEN = int(TARGET_HZ * WINDOW_SEC)

OUTDIR = SCRATCH + f"/mp-dataset/finetune_newcastle_{TARGET_HZ}hz_w{WINDOW_SEC}/"
userinputs=parse_args()
RELABEL = userinputs.relabel
is_Walmsley2020 = False
is_Willetts2018 = True
if RELABEL:
    if is_Walmsley2020:
        OUTDIR = os.path.join(OUTDIR, "relabel", "Walmsley2020")
    elif is_Willetts2018:
        OUTDIR = os.path.join(OUTDIR, "relabel", "Willetts2018")
    else:
        OUTDIR = os.path.join(OUTDIR, "relabel", "WillettsSpecific2018")



FILTERED_FILES = [os.path.join(FOLDER, fname) for fname in ["pat_inf.npy","p08b.npy", "p09b.npy", "p015b.npy"]]
COMBO = [os.path.join(FOLDER, fname) for fname in ["p08a.npy", "p09a.npy", "p015a.npy"]]
FILTERED_LABEL = []
##########################################################################


        
def main():
    X, Y, P = [], [], []
    total_samples = 0
    process_files()
    labeling()
    print("Create MTSSL Finetune dataset of Mendeley Data, (newcastle) dataset from study 'A multi-sensory dataset for the activities of daily living'...")
    cnt, cnt_nanlabel_window = 0, 0
    # Create the NPY directory if it does not exist    
    for filename in DATAFILES:
        # Replace 'filename.csv' with the actual file path
        data = np.load(filename, allow_pickle = True)
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # If DataFrame has more than 4 columns, keep only first 4 columns
        if len(df.columns) > 4:
            df = df.iloc[:, 0:4]
        df[3] =(df[3]).apply(label_sleep)
        filename_withoutEXT = os.path.splitext(filename)[0]
        pid = extract_number(filename_withoutEXT)
        total_samples += len(df)
        if len(df) >= WINDOW_LEN:
            for i in range(0, len(df), WINDOW_STEP_LEN):
                cnt += 1
                df_window = df.iloc[i : i + WINDOW_LEN]
                label_name = df_window[3].mode(dropna=True).values
                if label_name.shape[0] == 0:
                    cnt_nanlabel_window += 1
                    # print(f"Accelerometer data {i} : {i + WINDOW_LEN} has only unknown labels! {label_name}")
                    continue
                label_name = label_name[0]
                window = df_window[[0,1,2]]
                if not is_good_quality(window, WINDOW_LEN):
                    # print(f"Accelerometer data {i} : {i + WINDOW_LEN} has bad x,y,z quality!")
                    continue
                X.append(window.values.tolist())
                Y.append(label_name)
                P.append(pid)
        else:
            print("discarded!", filename, len(data))
    print(f"\nThere are {cnt_nanlabel_window}/{cnt} NaN label windows. Those are neglected from the final dataset.\n")

    X = np.asarray(X)
    # # Scale data from 8 bit to -4g, +4g
    # X = (X-127.5)*0.03137
    print('Max:', np.max(X))
    print('Min:', np.min(X))
    Y = np.asarray(Y)
    P = np.asarray(P)

    # downsample to 30 Hz
    X = resize(X, TARGET_WINDOW_LEN)
    # map newcastle labels to capture24 type
    if RELABEL:
        Y, idx_filter = label_mapto_capture24(Y,FILTERED_LABEL,is_Walmsley2020,is_Willetts2018)
        X = X[idx_filter,:,:]
        Y = Y[idx_filter]
        P = P[idx_filter]


    os.system(f"mkdir -p {OUTDIR}")
    np.save(os.path.join(OUTDIR, "X"), X)
    np.save(os.path.join(OUTDIR, "Y"), Y)
    np.save(os.path.join(OUTDIR, "pid"), P)

    print(f"Total samples: {total_samples}")
    print(f"Saved in {OUTDIR}")
    print("X shape:", X.shape)
    print("Y distribution:", len(set(Y)))
    print(pd.Series(Y).value_counts())
    print("User distribution:", len(np.unique(P)))
    print(pd.Series(P).value_counts())

if __name__ == "__main__":
    main()
