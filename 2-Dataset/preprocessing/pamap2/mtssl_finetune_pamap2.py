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
from utils import content2x_and_y, clean_up_label, resize, label_mapto_capture24, is_good_quality
import argparse

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--relabel", default=False, action='store_true', help='relabel from original labels to capture24 labels')
    args=parser.parse_args()
    return args

##########################################################################
# GLOBAL VARIABLES!!!

SCRATCH = os.environ['SCRATCH']
FOLDER = SCRATCH + '/mp-dataset/pamap2/PAMAP2_Dataset/'
data_path = FOLDER + "Protocol/"
PROTOCOL_FILES = glob.glob(data_path + "*.dat")
data_path = FOLDER + "Optional/"
OPTIONAL_FILES = glob.glob(data_path + "*.dat")
DATAFILES = PROTOCOL_FILES + OPTIONAL_FILES

DEVICE_HZ = 100  # Hz
WINDOW_SEC = 10  # seconds
WINDOW_OVERLAP_SEC = 5  # seconds
WINDOW_LEN = int(DEVICE_HZ * WINDOW_SEC)  # device ticks
WINDOW_OVERLAP_LEN = int(DEVICE_HZ * WINDOW_OVERLAP_SEC)  # device ticks
WINDOW_STEP_LEN = WINDOW_LEN - WINDOW_OVERLAP_LEN  # device ticks
WINDOW_TOL = 0.01  # 1%
TARGET_HZ = 30  # Hz
TARGET_WINDOW_LEN = int(TARGET_HZ * WINDOW_SEC)

OUTDIR = SCRATCH + f"/mp-dataset/finetune_pamap2_{TARGET_HZ}hz_w{WINDOW_SEC}/"
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


LABEL_NAMES = {
    1:"lying"
    ,2:"sitting"
    ,3:"standing"
    ,4:"walking"
    ,5:"running"
    ,6:"cycling"
    ,7:"Nordic walking"
    ,9:"watching TV"
    ,10:"computer work"
    ,11:"car driving"
    ,12:"ascending stairs"
    ,13:"descending stairs"
    ,16:"vacuum cleaning"
    ,17:"ironing"
    ,18:"folding laundry"
    ,19:"house cleaning"
    ,20:"playing soccer"
    ,24:"rope jumping"
}

FILTERED_LABEL = ["watching TV"]
##########################################################################

def main():

    X, Y, P = [], [], []

    print("Create MTSSL Finetune dataset of pamap2...")
    for file_path in tqdm(DATAFILES):
        # Read dat file contents
        p_id = int(file_path.split("/")[-1][-7:-4])
        datContent = [i.strip().split() for i in open(file_path).readlines()]
        datContent = np.array(datContent)

        # Extract timestamp, label, and accelerometer data.
        label_idx = 1
        timestamp_idx = 0
        x_idx = 4
        y_idx = 5
        z_idx = 6
        index_to_keep = [timestamp_idx, label_idx, x_idx, y_idx, z_idx]

        datContent = datContent[:, index_to_keep]
        datContent = datContent.astype(float)
        datContent = datContent[~np.isnan(datContent).any(axis=1)]

        # Create window, label, and id sets
        current_X, current_y = content2x_and_y(
            datContent, epoch_len=WINDOW_SEC, overlap=WINDOW_OVERLAP_SEC
        )
        current_X, current_y = clean_up_label(current_X, current_y, LABEL_NAMES)

        p_ids = np.full(
            shape=len(current_y), fill_value=p_id, dtype=np.int
        )
        # print(current_X.shape, current_y.shape, p_ids.shape)

        # Check the quality and add into the X, y, and pid dataset.
        for i, window in enumerate(current_X):
            window_df = pd.DataFrame(window)
            if not is_good_quality(window_df, WINDOW_LEN):
                print("data is bad quality.")
                continue
            else:
                X.append(window)
                Y.append(current_y[i])
                P.append(p_ids[i])

    X = np.asarray(X)
    Y = np.asarray(Y)
    P = np.asarray(P)

    Y = Y.flatten()
    X = X / constants.g  # convert to unit of g
    clip_value = 3
    X = np.clip(X, -clip_value, clip_value)

    # downsample to 30 Hz
    X = resize(X, TARGET_WINDOW_LEN)
    # map adl labels to capture24 type
    if RELABEL:
        Y, idx_filter = label_mapto_capture24(Y,FILTERED_LABEL,is_Walmsley2020,is_Willetts2018)
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
