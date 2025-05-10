"""
Create MTSSL finetuning data from the wristppg dataset.

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
from utils import resize, read_records, read_wristppg_wfdata, is_good_quality, label_mapto_capture24
import argparse

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--relabel", default=False, action='store_true', help='relabel from original labels to capture24 labels')
    args=parser.parse_args()
    return args

##########################################################################
# GLOBAL VARIABLES!!!

SCRATCH = os.environ['SCRATCH']
FOLDER = SCRATCH + '/mp-dataset/wristppg/wrist-ppg-during-exercise-1.0.0'
DATAFILES = SCRATCH + "/mp-dataset/wristppg/wrist-ppg-during-exercise-1.0.0/RECORDS"

DEVICE_HZ = 256  # Hz
WINDOW_SEC = 10  # seconds
WINDOW_OVERLAP_SEC = 5  # seconds
WINDOW_LEN = int(DEVICE_HZ * WINDOW_SEC)  # device ticks
WINDOW_OVERLAP_LEN = int(DEVICE_HZ * WINDOW_OVERLAP_SEC)  # device ticks
WINDOW_STEP_LEN = WINDOW_LEN - WINDOW_OVERLAP_LEN  # device ticks
WINDOW_TOL = 0.01  # 1%
TARGET_HZ = 30  # Hz
TARGET_WINDOW_LEN = int(TARGET_HZ * WINDOW_SEC)

OUTDIR = SCRATCH + f"/mp-dataset/finetune_wristppg_{TARGET_HZ}hz_w{WINDOW_SEC}/"
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

LABEL_NAMES = ['high_resistance_bike','low_resistance_bike','run','walk']

FILTERED_LABEL = []
##########################################################################

def main():

    X, Y, P = [], [], []
    total_samples = 0

    print("Create MTSSL Finetune dataset of wristppg...")
    cnt, cnt_nanlabel_window = 0, 0

    channels = ['wrist_low_noise_accelerometer_x',
                'wrist_low_noise_accelerometer_y',
                'wrist_low_noise_accelerometer_z',]
    records_lst = read_records(DATAFILES)

    for r in records_lst:
        filesetname = os.path.join(FOLDER,r)
        wf_dat = read_wristppg_wfdata(filesetname,channels)
        data = pd.DataFrame(wf_dat['p_signal'],columns=wf_dat['sig_name']).dropna()

        pid, label = wf_dat['record_name'].split('_',maxsplit=1)
        print(f"Processing... {r} {len(data)} samples")

        if len(data) >= WINDOW_LEN:
            for i in range(0, len(data), WINDOW_STEP_LEN):
                cnt += 1
                window = data.iloc[i : i + WINDOW_LEN]
                if not is_good_quality(window, WINDOW_LEN):
                    print(f"Accelerometer data {i} : {i + WINDOW_LEN} has bad x,y,z quality!")
                    continue
                X.append(window.values.tolist())
                Y.append(label)
                P.append(pid)
                total_samples += len(window.values.tolist())
        else:
            print("discarded!", r, len(data))

    X = np.asarray(X)
    Y = np.asarray(Y)
    P = np.asarray(P)

    # fixing unit to g
    X = X / 9.81
    # downsample to 30 Hz
    X = resize(X, TARGET_WINDOW_LEN)
    # map wristppg labels to capture24 type
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
