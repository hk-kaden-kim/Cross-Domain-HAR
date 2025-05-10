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
from utils import resize, label_mapto_capture24, is_good_quality
import argparse

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--relabel", default=False, action='store_true', help='relabel from original labels to capture24 labels')
    args=parser.parse_args()
    return args

##########################################################################
# GLOBAL VARIABLES!!!

DEVICE_HZ = 51.2  # Hz
WINDOW_SEC = 10  # seconds
WINDOW_OVERLAP_SEC = 5  # seconds
WINDOW_LEN = int(DEVICE_HZ * WINDOW_SEC)  # device ticks
WINDOW_OVERLAP_LEN = int(DEVICE_HZ * WINDOW_OVERLAP_SEC)  # device ticks
WINDOW_STEP_LEN = WINDOW_LEN - WINDOW_OVERLAP_LEN  # device ticks
WINDOW_TOL = 0.01  # 1%
TARGET_HZ = 30  # Hz
TARGET_WINDOW_LEN = int(TARGET_HZ * WINDOW_SEC)

SCRATCH = os.environ['SCRATCH']
DATAFILES = SCRATCH + '/mp-dataset/forth-trace/FORTH_TRACE_DATASET-master/'


OUTDIR = SCRATCH + f"/mp-dataset/finetune_forth-trace_{TARGET_HZ}hz_w{WINDOW_SEC}/"
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
1: "stand" 
,2: "sit" 
,3: "sit and talk" 
,4: "walk"
,5: "walk and talk" 
,6: "climb stairs (up/down)"
,7: "climb stairs (up/down) and talk" 
,8: "stand -> sit"
,9: "sit -> stand"
,10: "stand -> sit and talk"
,11: "sit and talk -> stand"
,12: "stand -> walk"
,13: "walk -> stand"
,14: "stand -> climb stairs (up/down), stand -> climb stairs (up/down) and talk"
,15: "climb stairs (up/down) -> walk"
,16: "climb stairs (up/down) and talk -> walk and talk"
}
# Columns in CSV files to consider
CSV_COLUMNS = ["accelerometer x", "accelerometer y", "accelerometer z", "Activity Label"]

# Parts and devices to consider
PARTS = [f"part{i}" for i in range(15)]
DEVICES = [f"dev{i}" for i in range(1, 6)]


FILTERED_LABEL = []
##########################################################################

CSV_COLUMNS = [
    'Device ID', 
    'accelerometer x', 
    'accelerometer y', 
    'accelerometer z', 
    'gyroscope x', 
    'gyroscope y', 
    'gyroscope z', 
    'magnetometer x', 
    'magnetometer y', 
    'magnetometer z', 
    'Timestamp', 
    'Activity Label'
]


SPECIFIC_COLUMNS = ["accelerometer x", "accelerometer y", "accelerometer z", "Activity Label"]

def process_chunk(df, X, Y, P, part):
    for i in range(0, len(df), WINDOW_STEP_LEN):
        window = df[i : i + WINDOW_LEN]

        if not is_good_quality(window, WINDOW_LEN):
            continue
        X.extend(df[SPECIFIC_COLUMNS[:-1]].values)
        Y.extend(df[SPECIFIC_COLUMNS[-1]].values)
        P.extend([int(part.replace('part', ''))] * len(df))


def combine_files():
    # Loop through all parts
    for part in PARTS:
        part_dir = os.path.join(DATAFILES, part)

        combined_data = None
        combined_path = os.path.join(part_dir, f"{part}_combined.npy")

        # Check if combined file already exists
        if os.path.exists(combined_path):
            continue

        # If combined file doesn't exist, create it
        for dev in DEVICES[:2]:
            dev_path = os.path.join(part_dir, f"{part}{dev}.csv")
            dev_df = pd.read_csv(dev_path, usecols=[1,2,3,11], names=SPECIFIC_COLUMNS)

            # If data already exists in combined_data, append new data. If not, just set new data
            if combined_data is not None:
                combined_data = np.concatenate((combined_data, dev_df.values))
            else:
                combined_data = dev_df.values

        # Save combined data to file
        np.save(combined_path, combined_data)

def main():
    X, Y, P = [], [], []
    total_samples = 0


    # Combine files first
    combine_files()

    # Then process each combined file
    # Loop through all parts
    for part in PARTS:
        part_dir = os.path.join(DATAFILES, part)
        combined_path = os.path.join(part_dir, f"{part}_combined.npy")

        combined_data = np.load(combined_path)

        # Process combined data
        num_rows = combined_data.shape[0]
        # print('num_rows',num_rows)
        pid = int(part.replace('part', ''))

        for i in range(0, num_rows, WINDOW_STEP_LEN):
            # end_row = min(i + WINDOW_LEN, num_rows)
            # print('WINDOW_LEN',WINDOW_LEN)
            # print('WINDOW_STEP_LEN',WINDOW_STEP_LEN)
            

            window = combined_data[i : i + WINDOW_LEN]
            # df = pd.DataFrame(combined_data[i:i + WINDOW_LEN], columns=SPECIFIC_COLUMNS)
            # total_samples += len(df)
            

            if not is_good_quality(window, WINDOW_LEN):
                continue

            # print(window,part)

            X.append(window[:,:-1])
            Y.append(window[0,-1])
            P.append(pid)
        
        # Remove combined file after processing
        os.remove(combined_path)
    
    X = np.asarray(X)
    Y = np.asarray(Y)
    vfunc = np.vectorize(LABEL_NAMES.get)
    Y = vfunc(Y)
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
    # np.save(os.path.join(OUTDIR, "session_id"), S)
    np.save(os.path.join(OUTDIR, "pid"), P)

    print(f"Total samples: {total_samples}")
    print(f"Saved in {OUTDIR}")
    print("X shape:", X.shape)
    # print("Y distribution:", len(np.unique(Y)))
    print("Y shape:", Y.shape)
    print("Y distribution:", len(set(Y)))
    print(pd.Series(Y).value_counts())
    # print(np.bincount(Y))
    print("P shape:", P.shape)
    print("User distribution:", len(set(P)))
    print(pd.Series(P).value_counts())
    
    # print(f"\n\nDelete cache output folder ... {CACHE_OUTPUT}")
    # # if os.path.exists(CACHE_OUTPUT):
    # #     rmtree(CACHE_OUTPUT)

if __name__ == "__main__":
    main()
