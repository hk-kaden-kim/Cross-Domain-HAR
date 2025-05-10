"""
Create MTSSL finetuning data from the harvardleo dataset.

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
FOLDER = SCRATCH + '/mp-dataset/harvardleo/'
DATAFILES = SCRATCH + "/mp-dataset/harvardleo/*.npy"

DEVICE_HZ = 256  # Hz
WINDOW_SEC = 10  # seconds
WINDOW_OVERLAP_SEC = 5  # seconds
WINDOW_LEN = int(DEVICE_HZ * WINDOW_SEC)  # device ticks
WINDOW_OVERLAP_LEN = int(DEVICE_HZ * WINDOW_OVERLAP_SEC)  # device ticks
WINDOW_STEP_LEN = WINDOW_LEN - WINDOW_OVERLAP_LEN  # device ticks
WINDOW_TOL = 0.01  # 1%
TARGET_HZ = 30  # Hz
TARGET_WINDOW_LEN = int(TARGET_HZ * WINDOW_SEC)

OUTDIR = SCRATCH + f"/mp-dataset/finetune_harvardleo_{TARGET_HZ}hz_w{WINDOW_SEC}/"
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
    1 : 'RELAX',
    2 : 'KEYBOARD_WRITING',
    3 : 'LAPTOP',
    4 : 'HANDWRITING',
    5 : 'HANDWASHING',
    6 : 'FACEWASHING',
    7 : 'TEETHBRUSH',
    8 : 'SWEEPING',
    9 : 'VACUUMING',
    10 : 'EATING',
    11 : 'DUSTING',
    12 : 'RUBBING',
    13 : 'DOWNSTAIRS',
    14 : 'WALKING',
    15 : 'WALKING_FAST',
    16 : 'UPSTAIRS_FAST',
    17 : 'UPSTAIRS'
    }



FILTERED_LABEL = []
##########################################################################

def combine_files():
    path = os.path.join(os.getenv("SCRATCH"),"mp-dataset/harvardleo")
    for i in range(1,9):
        npy_file = os.path.join(path,f'wrist_cobined_{i}.npy')
        if os.path.exists(npy_file):
            print("Combined file exists")
            continue
        else:
            print("Creating combined file")
            data = os.path.join(path, f'wrist_X_0{i}.tab')
            label = os.path.join(path, f'wrist_Y_0{i}.tab') 

            df1 = pd.read_csv(data, sep='\t')
            df1 = df1.drop(df1.columns[0], axis=1)
            df2 = pd.read_csv(label, sep='\t')

            df_combined = pd.concat([df1,df2],axis=1)
            df_combined['pid'] = i
            # save the combined data as .npy file
            np.save(npy_file, df_combined.to_numpy())
            print(f"Created .npy file: {npy_file}")

        



def main():

    X, Y, P = [], [], []
    total_samples = 0
    combine_files()
    print("Create MTSSL Finetune dataset of Harvard Dataverse, (HarvardLeo) dataset from study 'Daily Living Activity Recognition Using Wearable Devices: A Features-rich Dataset and a Novel Approach'...")
    cnt, cnt_nanlabel_window = 0, 0
    for filename in glob.glob(DATAFILES.format(FOLDER)):
        

        data = np.load(filename)
        df = pd.DataFrame(data, columns=["Accelerometer X", "Accelerometer Y", "Accelerometer Z", "label", "pid"])
        pid = df["pid"]
        total_samples += len(df)
        if len(df) >= WINDOW_LEN:
            for i in range(0, len(df), WINDOW_STEP_LEN):
                cnt += 1
                df_window = df.iloc[i : i + WINDOW_LEN]
                label_name = df_window['label'].replace(0.0,np.NaN).mode(dropna=True).values
                if label_name.shape[0] == 0:
                    cnt_nanlabel_window += 1
                    # print(f"Accelerometer data {i} : {i + WINDOW_LEN} has only unknown labels! {label_name}")
                    continue
                label_name = label_name[0]
                window = df_window[['Accelerometer X','Accelerometer Y','Accelerometer Z']]
                if not is_good_quality(window, WINDOW_LEN):
                    print(f"Accelerometer data {i} : {i + WINDOW_LEN} has bad x,y,z quality!")
                    continue
                X.append(window.values.tolist())
                Y.append(label_name)
                P.append(pid.values[0])
        else:
            print("discarded!", filename, len(data))
    print(f"\nThere are {cnt_nanlabel_window}/{cnt} NaN label windows. Those are neglected from the final dataset.\n")

    X = np.asarray(X)
    print('Max:', np.max(X))
    print('Min:', np.min(X))
    Y  = [LABEL_NAMES[int(y)] for y in Y]
    Y = np.asarray(Y)
    P = np.asarray(P)

    # downsample to 30 Hz
    X = resize(X, TARGET_WINDOW_LEN)
    # map harvardleo labels to capture24 type
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
