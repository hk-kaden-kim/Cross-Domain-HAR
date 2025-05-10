"""
Create MTSSL finetuning data from the paal dataset.

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
from scipy.spatial.distance import cdist
import argparse

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--relabel", default=False, action='store_true', help='relabel from original labels to capture24 labels')
    args=parser.parse_args()
    return args

##########################################################################
# GLOBAL VARIABLES!!!

SCRATCH = os.environ['SCRATCH']

FOLDER = os.path.join(SCRATCH, 'mp-dataset', 'paal', 'dataset')
DATAFILES = FOLDER + "/*.npy"

DEVICE_HZ = 32  # Hz
WINDOW_SEC = 10  # seconds
WINDOW_OVERLAP_SEC = 5  # seconds
WINDOW_LEN = int(DEVICE_HZ * WINDOW_SEC)  # device ticks
WINDOW_OVERLAP_LEN = int(DEVICE_HZ * WINDOW_OVERLAP_SEC)  # device ticks
WINDOW_STEP_LEN = WINDOW_LEN - WINDOW_OVERLAP_LEN  # device ticks
WINDOW_TOL = 0.01  # 1%
TARGET_HZ = 30  # Hz
TARGET_WINDOW_LEN = int(TARGET_HZ * WINDOW_SEC)

OUTDIR = SCRATCH + f"/mp-dataset/finetune_paal_{TARGET_HZ}hz_w{WINDOW_SEC}/"
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

participants = range(1,53)

HEADER = [ "Accelerometer X", "Accelerometer Y", "Accelerometer Z", "label"]

FILTERED_LABEL = []
##########################################################################

def combine_files():
    # get all CSV files in the directory
    folder_path = f'{SCRATCH}/mp-dataset/paal/dataset'
    all_files = os.listdir(folder_path)
    csv_files = [file for file in all_files if file.endswith('.csv')]

    # sort the files to ensure they are processed in the correct order
    csv_files.sort()

    # Check which participant IDs already have their npy.save file
    existing_participant_ids = set()
    for participant_id in range(1, 53):  # Assuming participant IDs range from 1 to 52
        participant_np_save_file = os.path.join(folder_path, f'participant_{participant_id}.npy')
        if os.path.exists(participant_np_save_file):
            existing_participant_ids.add(participant_id)

    # Remove CSV files corresponding to participants with existing np.save files
    csv_files = [file for file in csv_files if int(file.split('_')[-2]) not in existing_participant_ids]


    # initialize a dictionary to hold dataframes for each participant
    dfs = {}

    # iterate over CSV files
    for filename in csv_files:
        print(filename)
        # split filename on underscore
        parts = filename.split('_')

        # last two parts are participant ID and experiment number
        participant_id = int(parts[-2])
        experiment_number = int(parts[-1].split('.')[0])  # remove .csv extension
        print(participant_id, experiment_number)

        # activity label is all parts before participant ID
        activity_label = '_'.join(parts[:-2])
        print(activity_label)

        # read CSV file
        df_temp = pd.read_csv(os.path.join(folder_path, filename), names = ['Accelerometer X', 'Accelerometer Y', 'Accelerometer Z'])

        # add activity label
        df_temp['label'] = activity_label

        # if participant ID is already in dictionary, append to existing dataframe
        # else, add to dictionary with participant ID as key
        if participant_id in dfs:
            dfs[participant_id] = dfs[participant_id].append(df_temp)
        else:
            dfs[participant_id] = df_temp
        
        # iterate over the dictionary and save each dataframe as a numpy file
        for participant_id, df in dfs.items():
            # save as numpy file
            np.save(os.path.join(folder_path, f'participant_{participant_id}.npy'), df.to_numpy())


        
def main():

    X, Y, P = [], [], []
    total_samples = 0
    combine_files()
    print("Create MTSSL Finetune dataset of Mendeley Data, (paal) dataset from study 'A multi-sensory dataset for the activities of daily living'...")
    cnt, cnt_nanlabel_window = 0, 0
    for filename in glob.glob(DATAFILES.format(FOLDER)):
        # Remove the file extension
        data = np.load(filename, allow_pickle = True)
        df = pd.DataFrame(data, columns= HEADER)
        filename_withoutEXT = os.path.splitext(filename)[0]
        pid = int(filename_withoutEXT.split("_")[1])
        total_samples += len(df)
        if len(df) >= WINDOW_LEN:
            for i in range(0, len(df), WINDOW_STEP_LEN):
                cnt += 1
                df_window = df.iloc[i : i + WINDOW_LEN]
                label_name = df_window['label'].replace('none',np.NaN).mode(dropna=True).values
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
                P.append(pid)
        else:
            print("discarded!", filename, len(data))
    print(f"\nThere are {cnt_nanlabel_window}/{cnt} NaN label windows. Those are neglected from the final dataset.\n")

    X = np.asarray(X)
    # Scale data from milli-gravitational units (mG) to gravitational units (G)
    X = X * 0.015 # 0.015 G resolution
    print(f"X shape: {X.shape}")
    print('Max:', np.max(X))
    print('Min:', np.min(X))
    Y = np.asarray(Y)
    P = np.asarray(P)

    # downsample to 30 Hz
    X = resize(X, TARGET_WINDOW_LEN)
    # map paal labels to capture24 type
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
