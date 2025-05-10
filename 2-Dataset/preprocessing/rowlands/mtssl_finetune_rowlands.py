"""
Create MTSSL finetuning data from the adl dataset.

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
from scipy import constants
import pandas as pd
import os
import glob
from utils import parse_filename, fix_labels, resize, label_mapto_capture24, is_good_quality
import argparse

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--relabel", default=False, action='store_true', help='relabel from original labels to capture24 labels')
    args=parser.parse_args()
    return args

##########################################################################
# GLOBAL VARIABLES!!!

SCRATCH = os.environ['SCRATCH']
FOLDER = SCRATCH + '/mp-dataset/rowlands/'
data_path = FOLDER + "dataset/"
DATAFILES = glob.glob(data_path + "*.dat")

DEVICE_HZ = 80  # Hz
WINDOW_SEC = 10  # seconds
WINDOW_OVERLAP_SEC = 5  # seconds
WINDOW_LEN = int(DEVICE_HZ * WINDOW_SEC)  # device ticks
WINDOW_OVERLAP_LEN = int(DEVICE_HZ * WINDOW_OVERLAP_SEC)  # device ticks
WINDOW_STEP_LEN = WINDOW_LEN - WINDOW_OVERLAP_LEN  # device ticks
WINDOW_TOL = 0.01  # 1%
TARGET_HZ = 30  # Hz
TARGET_WINDOW_LEN = int(TARGET_HZ * WINDOW_SEC)

OUTDIR = SCRATCH + f"/mp-dataset/finetune_rowlands_{TARGET_HZ}hz_w{WINDOW_SEC}/"
userinputs=parse_args()
RELABEL = userinputs.relabel
is_Walmsley2020 = False
if RELABEL:
    if is_Walmsley2020:
        OUTDIR = os.path.join(OUTDIR, "relabel", "Walmsley2020")
    else:
        OUTDIR = os.path.join(OUTDIR, "relabel", "WillettsSpecific2018")


LABEL_NAMES = {
    1:"stand"
    ,2:"walk"
    ,4:"sit"
    ,5:"lie"
}

FILTERED_LABEL = []
##########################################################################

def main():

    X, Y, T, P, W = [], [], [], [], []
    
    print("Create MTSSL Finetune dataset of opportunity...")
    for file_path in tqdm(DATAFILES):
        data = pd.read_csv(file_path, parse_dates=["time"], index_col="time")

        part, wear = parse_filename(file_path)
        # Resample data
        period = int(round((1 / DEVICE_HZ) * 1000_000_000))
        data.resample(f"{period}N", origin="start").nearest(limit=1)

        for i in range(0, len(data), WINDOW_STEP_LEN):
            w = data.iloc[i : i + WINDOW_LEN]
            if not is_good_quality(w, WINDOW_LEN, WINDOW_SEC, WINDOW_TOL):
                print("data is bad quality.")
                continue
            else:
                t = w.index[0].to_datetime64()
                x = w[["x", "y", "z"]].values
                y = w["label"][0]
                y = fix_labels(y)
                
                X.append(x)
                Y.append(y)
                T.append(t)
                P.append(part)
                W.append(wear)

    X = np.asarray(X)
    Y = np.asarray(Y)
    T = np.asarray(T)
    P = np.asarray(P)
    W = np.asarray(W)


    # downsample to 30 Hz
    X = resize(X, TARGET_WINDOW_LEN)
    # map adl labels to capture24 type
    if RELABEL:
        Y, idx_filter = label_mapto_capture24(Y,FILTERED_LABEL,is_Walmsley2020)
        X = X[idx_filter,:,:]
        Y = Y[idx_filter]
        P = P[idx_filter]

    df = pd.DataFrame({"y": Y, "pid": P})
    print(df.groupby("pid")["y"].unique())

    os.system(f"mkdir -p {OUTDIR}")
    np.save(os.path.join(OUTDIR, "X"), X)
    np.save(os.path.join(OUTDIR, "Y"), Y)
    np.save(os.path.join(OUTDIR, "pid"), P)

    print(f"Saved in {OUTDIR}")
    print("X shape:", X.shape)
    print("Y distribution:", len(set(Y)))
    print(pd.Series(Y).value_counts())
    print("User distribution:", len(set(P)), P.shape)
    print(pd.Series(P).value_counts())

if __name__ == "__main__":
    main()
