"""
Create MTSSL finetuning data from the realworld dataset.

Inputs
-------------------

Outputs
-------------------
    X.npy
    y.npy
    session_id.npy
    pid.npy
"""

import glob
import os
from shutil import rmtree
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from utils import resize, label_mapto_capture24, intermediate_preprocess, is_numpy_array_good_quality
import argparse

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--relabel", default=False, action='store_true', help='relabel from original labels to capture24 labels')
    args=parser.parse_args()
    return args

##########################################################################
# GLOBAL VARIABLES!!!

DEVICE_HZ = 50  # Hz
WINDOW_SEC = 10  # seconds
WINDOW_OVERLAP_SEC = 5  # seconds
WINDOW_LEN = int(DEVICE_HZ * WINDOW_SEC)  # device ticks
WINDOW_OVERLAP_LEN = int(DEVICE_HZ * WINDOW_OVERLAP_SEC)  # device ticks
WINDOW_STEP_LEN = WINDOW_LEN - WINDOW_OVERLAP_LEN  # device ticks
WINDOW_TOL = 0.01  # 1%
TARGET_HZ = 30  # Hz
TARGET_WINDOW_LEN = int(TARGET_HZ * WINDOW_SEC)

SCRATCH = os.environ['SCRATCH']
ORIGIN_DATASET = SCRATCH + '/mp-dataset/realworld'
CACHE_OUTPUT = ORIGIN_DATASET + "/__imu__"
DATAFILES = CACHE_OUTPUT + "/*.npy"

OUTDIR = SCRATCH + f"/mp-dataset/finetune_realworld_{TARGET_HZ}hz_w{WINDOW_SEC}/"
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


PID = list(range(1, 16))
LABEL_NAMES = [
    "jumping",
    "climbingup",
    "climbingdown",
    "lying",
    "running",
    "sitting",
    "standing",
    "walking",
]
BODY_PARTS = ["chest", "forearm", "head", "shin", "thigh", "upperarm", "waist"]
start_ind = BODY_PARTS.index("forearm") * 3

FILTERED_LABEL = ["watching TV"]
##########################################################################


def main():

    # Original dataset filtering and reorganizing process
    print("-------------------")
    intermediate_preprocess(ORIGIN_DATASET, CACHE_OUTPUT, PID, LABEL_NAMES, BODY_PARTS)
    print("-------------------")
    
    # Create X, Y, S(Session), P npy files
    X, Y, S, P = [], [], [], []
    total_samples = 0
    for datafile in tqdm(glob.glob(DATAFILES)):
        pid, sess_id, class_name, _ = datafile.split("/")[-1].split(".")
        data = np.load(datafile)[
            :, start_ind : start_ind + 3
        ]  # data corresponding to forearm
        total_samples += len(data)
        for i in range(0, len(data), WINDOW_STEP_LEN):
            window = data[i : i + WINDOW_LEN]
            if not is_numpy_array_good_quality(window, WINDOW_LEN):
                continue
            X.append(window)
            Y.append(class_name)
            S.append(sess_id)
            P.append(pid)

    X = np.asarray(X)
    Y = np.asarray(Y)
    S = np.asarray(S)
    P = np.asarray(P)

    # fixing unit to g
    X = X / 9.81
    # downsample to 30 Hz
    X = resize(X, TARGET_WINDOW_LEN)
    # map realworld labels to capture24 type
    if RELABEL:
        Y, idx_filter = label_mapto_capture24(Y,FILTERED_LABEL,is_Walmsley2020,is_Willetts2018)
        X = X[idx_filter,:,:]
        Y = Y[idx_filter]
        P = P[idx_filter]

    os.system(f"mkdir -p {OUTDIR}")
    np.save(os.path.join(OUTDIR, "X"), X)
    np.save(os.path.join(OUTDIR, "Y"), Y)
    np.save(os.path.join(OUTDIR, "session_id"), S)
    np.save(os.path.join(OUTDIR, "pid"), P)

    print(f"Total samples: {total_samples}")
    print(f"Saved in {OUTDIR}")
    print("X shape:", X.shape)
    print("Y distribution:", len(set(Y)))
    print(pd.Series(Y).value_counts())
    print("User distribution:", len(set(P)))
    print(pd.Series(P).value_counts())
    
    print(f"\n\nDelete cache output folder ... {CACHE_OUTPUT}")
    if os.path.exists(CACHE_OUTPUT):
        rmtree(CACHE_OUTPUT)

if __name__ == "__main__":
    main()
