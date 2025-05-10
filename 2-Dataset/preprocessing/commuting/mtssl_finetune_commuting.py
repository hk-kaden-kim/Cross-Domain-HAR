"""
Create MTSSL finetuning data from the commuting dataset.

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
from datetime import datetime

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--relabel", default=False, action='store_true', help='relabel from original labels to capture24 labels')
    args=parser.parse_args()
    return args

##########################################################################
# GLOBAL VARIABLES!!!

SCRATCH = os.environ['SCRATCH']

FOLDER = os.path.join(SCRATCH, 'mp-dataset', 'commuting')
DATAFILES = FOLDER + "/*.npy"


DEVICE_HZ = 20  # Hz
WINDOW_SEC = 10  # seconds
WINDOW_OVERLAP_SEC = 5  # seconds
WINDOW_LEN = int(DEVICE_HZ * WINDOW_SEC)  # device ticks
WINDOW_OVERLAP_LEN = int(DEVICE_HZ * WINDOW_OVERLAP_SEC)  # device ticks
WINDOW_STEP_LEN = WINDOW_LEN - WINDOW_OVERLAP_LEN  # device ticks
WINDOW_TOL = 0.01  # 1%
TARGET_HZ = 30  # Hz
TARGET_WINDOW_LEN = int(TARGET_HZ * WINDOW_SEC)

OUTDIR = SCRATCH + f"/mp-dataset/finetune_commuting_{TARGET_HZ}hz_w{WINDOW_SEC}/"
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

HEADER = ["timestamp","Accelerometer X", "Accelerometer Y", "Accelerometer Z", "Light", "Button", "Temprature", "label"]

FILTERED_LABEL = []

##########################################################################

def combine_files():



    def get_activity(row, activity_df):
        timestamp = (datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S:%f")).time()
        for i in activity_df.index:
            if activity_df["start"][i] <= timestamp <= activity_df["end"][i]:
                return activity_df["activity"][i]
            else:
                return "none"
        return "Other"

    def process_files(directory, output_file):
        output_path_existing = os.path.join(FOLDER, output_file)
        output_path_non_existing = os.path.join(directory, output_file)

    
        if os.path.isfile(output_path_existing):
            print(f"File {output_path_existing} exists, it will not be re-created.")
        else:
            print(f"File {output_path_non_existing} does not exist, it will be created.")

            # Initialize an empty DataFrame to hold all data
            all_data = pd.DataFrame()

            # Iterate over each file in the directory
            for filename in os.listdir(directory):
                if filename.endswith(".csv"):
                    # Read each CSV file
                    csv_data = pd.read_csv(os.path.join(directory, filename), skiprows=100, parse_dates=[0], names=["timestamp", "MEMS x", "MEMS y", "MEMS z", "Lux", "Event", "Temp"])
                    # Read corresponding txt file
                    txt_file = filename.replace('.csv', '.txt')
                    activity_df = pd.read_csv(os.path.join(directory, txt_file), sep='\t', skiprows=1, names=["activity","start", "end"])

                    activity_df["start"] = pd.to_datetime(activity_df["start"]).dt.time
                    activity_df["end"] = pd.to_datetime(activity_df["end"]).dt.time

                    # Apply the labels
                    csv_data["activity"] = csv_data.apply(lambda row: get_activity(row, activity_df), axis=1)
                    
                    # Append the data from this CSV to all_data
                    all_data = all_data.append(csv_data, ignore_index=True)

            # Save to .npy file
            np.save(os.path.join(FOLDER, output_file), all_data.values)

    # Usage
    directory1 = FOLDER + '/user1/'
    process_files(directory1, 'output_1.npy')

    directory2 = FOLDER + '/user2/'
    process_files(directory2, 'output_2.npy')


        
def main():

    X, Y, P = [], [], []
    total_samples = 0
    combine_files()
    print("Create MTSSL Finetune dataset of Commuting Data")
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
    # X = X * 0.015 # 0.015 G resolution
    print('Max:', np.max(X))
    print('Min:', np.min(X))
    Y = np.asarray(Y)
    P = np.asarray(P)

    # downsample to 30 Hz
    X = resize(X, TARGET_WINDOW_LEN)
    # map commuting labels to capture24 type
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
