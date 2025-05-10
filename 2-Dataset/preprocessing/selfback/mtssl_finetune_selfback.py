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
from scipy import constants

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--relabel", default=False, action='store_true', help='relabel from original labels to capture24 labels')
    args=parser.parse_args()
    return args

##########################################################################
# GLOBAL VARIABLES!!!

DEVICE_HZ = 100  # Hz
WINDOW_SEC = 10  # seconds
WINDOW_OVERLAP_SEC = 5  # seconds
WINDOW_LEN = int(DEVICE_HZ * WINDOW_SEC)  # device ticks
WINDOW_OVERLAP_LEN = int(DEVICE_HZ * WINDOW_OVERLAP_SEC)  # device ticks
WINDOW_STEP_LEN = WINDOW_LEN - WINDOW_OVERLAP_LEN  # device ticks
WINDOW_TOL = 0.01  # 1%
TARGET_HZ = 30  # Hz
TARGET_WINDOW_LEN = int(TARGET_HZ * WINDOW_SEC)

SCRATCH = os.environ['SCRATCH']
# ROOT_DIR = SCRATCH + '/mp-dataset/selfback/selfBACK/w/'
ROOT_DIR = SCRATCH + '/mp-dataset/selfback/w/'

OUTDIR = SCRATCH + f"/mp-dataset/finetune_selfback_{TARGET_HZ}hz_w{WINDOW_SEC}/"
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
    'upstairs': 'Walking Upstairs',
    'downstairs': 'Walking Downstairs',
    'walk_slow': 'Walking in slow pace',
    'walk_mod': 'Walking in medium pace',
    'walk_fast': 'Walking in fast pace',
    'jogging': 'Jogging',
    'standing': 'Standing',
    'sitting': 'Sitting',
    'lying': 'Lying'
}
# Columns in CSV files to consider
COLUMNS = ['time', 'x', 'y', 'z']




FILTERED_LABEL = []
##########################################################################




def combine_files():
    # Iterate over each activity directory
    for activity_folder in os.listdir(ROOT_DIR):
        if not activity_folder.endswith("_combined.npy"):
            if activity_folder.startswith("._"):
                continue
            print(f"Processing activity: {activity_folder}")

            # Initialize empty DataFrame to hold all data for this activity
            activity_data = pd.DataFrame()
            
            # Define column types
            col_types = {
                 'time': 'object',
                 'x': 'float64',
                'y': 'float64',
                'z': 'float64',
            }
            # Iterate over each participant file in the activity directory
            for participant_file in glob.glob(os.path.join(ROOT_DIR, activity_folder, "*.csv")):
                # Skip files starting with ._
                if os.path.basename(participant_file).startswith("._"):
                    continue

                # Get participant id from filename (remove leading zeros)
                participant_id = int(os.path.basename(participant_file).split(".")[0].lstrip("0"))

                # Load participant data
                participant_data = pd.read_csv(participant_file, header=0, names=COLUMNS, dtype=col_types)


                # Add participant id and activity to participant data
                participant_data['pid'] = participant_id
                participant_data['activity'] = activity_folder
                participant_data['activity'] = participant_data['activity'].map(LABEL_NAMES)

                # Append to activity_data DataFrame
                activity_data = activity_data.append(participant_data, ignore_index=True)
                # print(activity_data)

        # Save the combined data for this activity to a .npy file
        combined_path = os.path.join(ROOT_DIR, f"{activity_folder}_combined.npy")
        np.save(combined_path, activity_data.to_numpy())

print("Done!")

def main():
    userinputs=parse_args()
    RELABEL = userinputs.relabel
    # Initialize lists for windows
    X, Y, P = [], [], []
    combine_files()
    # Iterate over each combined npy file
    for combined_file in os.listdir(ROOT_DIR):
        if combined_file.endswith("_combined.npy"):
            activity = combined_file.split("_")[0]

            print(f"Processing combined data for activity: {activity}")

            # Load combined data
            combined_data = np.load(os.path.join(ROOT_DIR, combined_file), allow_pickle=True)
           
            

            # Apply sliding window to combined data
            num_rows = combined_data.shape[0]
            for i in range(0, num_rows, WINDOW_STEP_LEN):
                window = combined_data[i : i + WINDOW_LEN]
   
                if not is_good_quality(window, WINDOW_LEN):
                    continue

                X.append(window[:,1:-2])
                Y.append(window[0,-1])
                P.append(window[0,-2])
            os.remove(os.path.join(ROOT_DIR, combined_file))
    
    X = np.asarray(X)
    Y = np.asarray(Y)
    P = np.asarray(P)
    X = X / constants.g
    clip_value = 3
    X = np.clip(X, -clip_value, clip_value)
   

    
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

    # print(f"Total samples: {total_samples}")
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
    


if __name__ == "__main__":
    main()
