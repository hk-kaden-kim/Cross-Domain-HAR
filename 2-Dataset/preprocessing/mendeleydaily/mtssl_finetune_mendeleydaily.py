"""
Create MTSSL finetuning data from the mendeleydaily dataset.

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

FOLDER = os.path.join(SCRATCH, 'mp-dataset', 'mendeleydaily', 'A multi-sensory dataset for the activities of daily living')
DATAFILES = FOLDER + "/*.npy"

DEVICE_HZ = 33  # Hz
WINDOW_SEC = 10  # seconds
WINDOW_OVERLAP_SEC = 5  # seconds
WINDOW_LEN = int(DEVICE_HZ * WINDOW_SEC)  # device ticks
WINDOW_OVERLAP_LEN = int(DEVICE_HZ * WINDOW_OVERLAP_SEC)  # device ticks
WINDOW_STEP_LEN = WINDOW_LEN - WINDOW_OVERLAP_LEN  # device ticks
WINDOW_TOL = 0.01  # 1%
TARGET_HZ = 30  # Hz
TARGET_WINDOW_LEN = int(TARGET_HZ * WINDOW_SEC)

OUTDIR = SCRATCH + f"/mp-dataset/finetune_mendeleydaily_{TARGET_HZ}hz_w{WINDOW_SEC}/"
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


HEADER = ["qags", "time_stamp", "quaternion1", "quaternion2", "quaternion3", "quaternion4", "Accelerometer X", "Accelerometer Y", "Accelerometer Z", "angular_velocity_x", "angular_velocity_y", "angular_velocity_z","label"]

FILTERED_LABEL = []
##########################################################################

def combine_files():
    volunteers = [f for f in glob.glob(os.path.join(FOLDER, "volunteer_*"))if os.path.isdir(f) and not f.endswith('.npy')]
    sensors = ["lla", "rla"]
    sensor_column_names = ["qags", "time_stamp", "quaternion1", "quaternion2", "quaternion3", "quaternion4", "acceleration_x", "acceleration_y", "acceleration_z", "angular_velocity_x", "angular_velocity_y", "angular_velocity_z"]

    for i, vol in enumerate(volunteers):
        # Check if the npy file already exists - fixed
        if os.path.exists(f'{vol}.npy'):
            print(f'{vol}.npy already exists. Skipping...')
            continue
        else:
            print('Creating' + f'{vol}.npy ...')
            annotation_file = "annotations.CSV" if os.path.exists(os.path.join(vol, "annotations.CSV")) else "annotation.CSV"
            T = pd.read_csv(os.path.join(vol, annotation_file))
            start_instants = T[T['Start/End'] == "Start"].index
            start_time = T.loc[start_instants, 'Time [msec]'].values
            end_instants = T[T['Start/End'] == "End"].index
            end_time = T.loc[end_instants, 'Time [msec]'].values.astype(int)
            activity_labels = T.loc[start_instants, 'BothArmsLabel'].tolist()

            combined_sensor_data = []
            combined_labels = []
            for j, sens in enumerate(sensors):
                data_table = pd.read_csv(os.path.join(vol, "IMUs", sens + ".csv"),names=sensor_column_names)
                stamps = data_table['time_stamp'].values.astype(int)
                abs_diff_matrix_start = np.abs(stamps[:, np.newaxis] - start_time)
                closestIndex_start = abs_diff_matrix_start.argmin(axis=0)

                abs_diff_matrix_end = np.abs(stamps[:, np.newaxis] - end_time)
                closestIndex_end = abs_diff_matrix_end.argmin(axis=0)

                data_table[sens + '_label'] = 'none'
                for k, (start_idx, end_idx) in enumerate(zip(closestIndex_start, closestIndex_end)):
                    data_table.loc[start_idx:end_idx, sens + '_label'] = activity_labels[k]

                combined_sensor_data.append(data_table.iloc[:, :-1])  # Get all sensor data columns except the label
                combined_labels.append(data_table[sens + '_label'])

            sensor_df = pd.concat(combined_sensor_data, axis=0)  # Vertical concatenation
            label_df = pd.concat(combined_labels, axis=0)  # Vertical concatenation

            combined_df = pd.concat([sensor_df, label_df], axis=1)
            # Saving path - fixed
            np.save(f'{vol}' + '.npy', combined_df.values)

    

    
        



def main():

    X, Y, P = [], [], []
    total_samples = 0
    combine_files()
    print("Create MTSSL Finetune dataset of Mendeley Data, (mendeleydaily) dataset from study 'A multi-sensory dataset for the activities of daily living'...")
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
    X = X / 10000 # 0.1 mG resolution to G
    Y = np.asarray(Y)
    P = np.asarray(P)

    # downsample to 30 Hz
    X = resize(X, TARGET_WINDOW_LEN)
    # map mendeleydaily labels to capture24 type
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
