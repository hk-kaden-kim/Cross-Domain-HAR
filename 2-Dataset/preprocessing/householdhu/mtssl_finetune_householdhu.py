"""
Create MTSSL finetuning data from the householdhu dataset.

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
import gc
import scipy.io
import glob
from utils import resize, is_good_quality, label_mapto_capture24
import argparse

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--relabel", default=False, action='store_true', help='relabel from original labels to capture24 labels')
    args=parser.parse_args()
    return args
#########################################################################
# GLOBAL VARIABLES!!!

SCRATCH = os.environ['SCRATCH']
FOLDER = SCRATCH + '/mp-dataset/householdhu/Multimodal_fine_grained_human_activity_data/data_raw/'
DATAFILES = FOLDER + "*.npy"

DEVICE_HZ = 235  # Hz
WINDOW_SEC = 10  # seconds
WINDOW_OVERLAP_SEC = 5  # seconds
WINDOW_LEN = int(DEVICE_HZ * WINDOW_SEC)  # device ticks
WINDOW_OVERLAP_LEN = int(DEVICE_HZ * WINDOW_OVERLAP_SEC)  # device ticks
WINDOW_STEP_LEN = WINDOW_LEN - WINDOW_OVERLAP_LEN  # device ticks
WINDOW_TOL = 0.01  # 1%
TARGET_HZ = 30  # Hz
TARGET_WINDOW_LEN = int(TARGET_HZ * WINDOW_SEC)

OUTDIR = SCRATCH + f"/mp-dataset/finetune_householdhu_{TARGET_HZ}hz_w{WINDOW_SEC}/"
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
                1: "Keyboard typing",
                2: "Using mouse",
                3: "Handwriting",
                4: "Cutting vegetables",
                5: "Stir-frying vegetables",
                6: "Wiping the table",
                7: "Sweeping floor",
                8: "Using vacuum to vacuum",
                9: "Open and close drawer",
                10: "None Activity"
              }


FILTERED_LABEL = []
##########################################################################

def combine_files():
    path = FOLDER
    # Path to your .mat file
    for i in range(1, 7):
        
        npy_file = os.path.join(path,f'combined_{i}.npy')
        print(npy_file)
        if os.path.exists(npy_file): 
            print("Combined file exists")
            continue
        else:
            print(f"Creation of combined_{i}.npy")
            mat_file_path1 = os.path.join(path,f'p{i}l1.mat')
            mat_file_path2 = os.path.join(path,f'p{i}l2.mat')

        # Load the .mat file
        mat1 = scipy.io.loadmat(mat_file_path1)
        mat2 = scipy.io.loadmat(mat_file_path2)

        # Load each array in the .mat file into a separate DataFrame
        df_clean1 = pd.DataFrame(mat1['clean_cleaned'])
        df_cook1 = pd.DataFrame(mat1['cook_cleaned'])
        df_vac1 = pd.DataFrame(mat1['vac_cleaned'])
        df_work1 = pd.DataFrame(mat1['work_cleaned'])
        df_clean2 = pd.DataFrame(mat2['clean_cleaned'])
        df_cook2 = pd.DataFrame(mat2['cook_cleaned'])
        df_vac2 = pd.DataFrame(mat2['vac_cleaned'])
        df_work2 = pd.DataFrame(mat2['work_cleaned'])
        print(df_work2.shape)

        # Select only the required columns (1, 6, 7, 8) from each dataframe
        # Note: Python uses 0-based indexing, so we subtract 1 from each index
        df_clean1 = df_clean1.iloc[:, [0, 6, 7, 8]]
        df_cook1 = df_cook1.iloc[:, [0, 6, 7, 8]]
        df_vac1 = df_vac1.iloc[:, [0, 6, 7, 8]]
        df_work1 = df_work1.iloc[:, [0, 6, 7, 8]]
        df_clean2 = df_clean2.iloc[:, [0, 6, 7, 8]]
        df_cook2 = df_cook2.iloc[:, [0, 6, 7, 8]]
        df_vac2 = df_vac2.iloc[:, [0, 6, 7, 8]]
        df_work2 = df_work2.iloc[:, [0, 6, 7, 8]]

        # Concatenate all dataframes into one
        df_combined = pd.concat([df_clean1, df_cook1, df_vac1, df_work1, df_clean2, df_cook2, df_vac2, df_work2], ignore_index=True)

        # Convert DataFrame to numpy array
        combined_array = df_combined.to_numpy()

        # Build output file path
        output_file_path = os.path.join(path, f'combined_{i}.npy')

        # Save numpy array to a .npy file
        np.save(output_file_path, combined_array)
        # explicitly delete DataFrames to free memory
        del df_clean1, df_cook1, df_vac1, df_work1, df_clean2, df_cook2, df_vac2, df_work2, df_combined, combined_array
        gc.collect()  # call garbage collector
        

    


def main():

    X, Y, P = [], [], []
    total_samples = 0
    combine_files()
    print("Create MTSSL Finetune dataset of householdhu...")
    cnt, cnt_nanlabel_window = 0, 0
    for filename in glob.glob(DATAFILES.format(FOLDER)):
        print(filename)
        pid = filename.split("combined_")[1]

        data = pd.DataFrame(np.load(filename), columns=["label","Accelerometer X", "Accelerometer Y", "Accelerometer Z"])
        total_samples += len(data)
        if len(data) >= WINDOW_LEN:
            for i in range(0, len(data), WINDOW_STEP_LEN):
                cnt += 1
                df_window = data.iloc[i : i + WINDOW_LEN]
                label_name = df_window['label'].replace(10.0,np.NaN).mode(dropna=True).values
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
    X = np.clip(X, -3, 3)

    print('Max:', np.max(X))
    print('Min:', np.min(X))
    Y  = [LABEL_NAMES[int(y)] for y in Y]
    Y = np.asarray(Y)
    P = np.asarray(P)

    # downsample to 30 Hz
    X = resize(X, TARGET_WINDOW_LEN)
    # map gotov labels to capture24 type
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
