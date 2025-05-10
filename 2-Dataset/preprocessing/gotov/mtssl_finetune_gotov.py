"""
Create MTSSL finetuning data from the gotov dataset.

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
from utils import resize, is_good_quality, label_mapto_capture24
import argparse

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--relabel", default=False, action='store_true', help='relabel from original labels to capture24 labels')
    args=parser.parse_args()
    return args

##########################################################################
# GLOBAL VARIABLES!!!

SCRATCH = os.environ['SCRATCH']
FOLDER = SCRATCH + '/mp-dataset/gotov/'
DATAFILES = SCRATCH + "/mp-dataset/gotov/{}/*wrist.csv"

DEVICE_HZ = 83  # Hz
WINDOW_SEC = 10  # seconds
WINDOW_OVERLAP_SEC = 5  # seconds
WINDOW_LEN = int(DEVICE_HZ * WINDOW_SEC)  # device ticks
WINDOW_OVERLAP_LEN = int(DEVICE_HZ * WINDOW_OVERLAP_SEC)  # device ticks
WINDOW_STEP_LEN = WINDOW_LEN - WINDOW_OVERLAP_LEN  # device ticks
WINDOW_TOL = 0.01  # 1%
TARGET_HZ = 30  # Hz
TARGET_WINDOW_LEN = int(TARGET_HZ * WINDOW_SEC)

OUTDIR = SCRATCH + f"/mp-dataset/finetune_gotov_{TARGET_HZ}hz_w{WINDOW_SEC}/"
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

LABEL_NAMES = [
     'syncJumping'
    , 'standing'
    , 'step'
    , 'lyingDownLeft'
    , 'lyingDownRight'
    , 'sittingSofa'
    , 'sittingCouch'
    , 'sittingChair'
    , 'walkingStairsUp'
    , 'dishwashing'
    , 'stakingShelves'
    , 'vacuumCleaning'
    , 'walkingSlow'
    , 'walkingNormal'
    , 'walkingFast'
    , 'cycling'
    ]

FILTERED_LABEL = []
##########################################################################

def main():

    X, Y, P = [], [], []
    total_samples = 0

    print("Create MTSSL Finetune dataset of gotov...")
    cnt, cnt_nanlabel_window = 0, 0
    for folder_name in os.listdir(FOLDER):
        for filename in glob.glob(DATAFILES.format(folder_name)):
            print(filename)
            pid = folder_name.split("GOTOV")[1]

            data = pd.read_csv(
                                    filename,
                                    names=["index", "time", "x", "y", "z", "annotation"],
                                    dtype={
                                        "x": "f4",
                                        "y": "f4",
                                        "z": "f4",
                                        "annotation": "string",
                                    },
                                    sep=",",
                                    na_filter=False,
                                    keep_default_na=False,
                                    skiprows=1,
                                )

            total_samples += len(data)
            if len(data) >= WINDOW_LEN:
                for i in range(0, len(data), WINDOW_STEP_LEN):
                    cnt += 1
                    df_window = data.iloc[i : i + WINDOW_LEN]
                    label_name = df_window['annotation'].replace('NA',np.NaN).mode(dropna=True).values
                    if label_name.shape[0] == 0:
                        cnt_nanlabel_window += 1
                        # print(f"Accelerometer data {i} : {i + WINDOW_LEN} has only unknown labels! {label_name}")
                        continue
                    label_name = label_name[0]
                    window = df_window[['x','y','z']]
                    if not is_good_quality(window, WINDOW_LEN):
                        print(f"Accelerometer data {i} : {i + WINDOW_LEN} has bad x,y,z quality!")
                        continue
                    X.append(window.values.tolist())
                    Y.append(label_name)
                    P.append(pid)
            else:
                print("discarded!", folder_name, filename, len(data))
    print(f"\nThere are {cnt_nanlabel_window}/{cnt} NaN label windows. Those are neglected from the final dataset.\n")

    X = np.asarray(X)
    print('Max:', np.max(X))
    print('Min:', np.min(X))
    Y = np.asarray(Y)
    P = np.asarray(P)

    # downsample to 30 Hz
    X = resize(X, TARGET_WINDOW_LEN)
    # map gotov labels to capture24 type
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
    print("User distribution:", len(set(P)))
    print(pd.Series(P).value_counts())

if __name__ == "__main__":
    main()
