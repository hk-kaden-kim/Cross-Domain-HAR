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
import pandas as pd
import os
import glob
from utils import resize, label_mapto_capture24, is_good_quality
import argparse

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--relabel", default=False, action='store_true', help='relabel from original labels to capture24 labels')
    args=parser.parse_args()
    return args

##########################################################################
# GLOBAL VARIABLES!!!

SCRATCH = os.environ['SCRATCH']
FOLDER = SCRATCH + '/mp-dataset/adl/HMP_Dataset/'
DATAFILES = SCRATCH + "/mp-dataset/adl/HMP_Dataset/{}/*.txt"

DEVICE_HZ = 32  # Hz
WINDOW_SEC = 10  # seconds
WINDOW_OVERLAP_SEC = 5  # seconds
WINDOW_LEN = int(DEVICE_HZ * WINDOW_SEC)  # device ticks
WINDOW_OVERLAP_LEN = int(DEVICE_HZ * WINDOW_OVERLAP_SEC)  # device ticks
WINDOW_STEP_LEN = WINDOW_LEN - WINDOW_OVERLAP_LEN  # device ticks
WINDOW_TOL = 0.01  # 1%
TARGET_HZ = 30  # Hz
TARGET_WINDOW_LEN = int(TARGET_HZ * WINDOW_SEC)

OUTDIR = SCRATCH + f"/mp-dataset/finetune_adl_{TARGET_HZ}hz_w{WINDOW_SEC}/"
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

LABEL_NAMES = [
    "brush_teeth",
    "climb_stairs",
    "comb_hair",
    "descend_stairs",
    "drink_glass",
    "eat_meat",
    "eat_soup",
    "getup_bed",
    "liedown_bed",
    "pour_water",
    "sitdown_chair",
    "standup_chair",
    "use_telephone",
    "walk",
]

FILTERED_LABEL = []
##########################################################################

def main():

    X, Y, P = [], [], []
    total_samples = 0

    print("Create MTSSL Finetune dataset of adl...")
    for folder_name in os.listdir(FOLDER):
        print(folder_name)
        if folder_name.lower() in LABEL_NAMES or folder_name.endswith("MODEL"):
            print(folder_name)
            label_name = (
                folder_name[:-6]
                if folder_name.endswith("MODEL")
                else folder_name
            )
            label_name = label_name.lower()
            for filename in tqdm(glob.glob(DATAFILES.format(folder_name))):
                pid = filename.split(".")[-2].split("-")[-1]
                data = pd.read_csv(filename, delimiter=" ", header=None)
                data = data / 63 * 3 - 1.5 # TODO : Check the reason of scailing and offset of values

                total_samples += len(data)
                if len(data) >= WINDOW_LEN:
                    for i in range(0, len(data), WINDOW_STEP_LEN):
                        window = data.iloc[i : i + WINDOW_LEN]
                        if not is_good_quality(window, WINDOW_LEN):
                            continue
                        X.append(window.values.tolist())
                        Y.append(label_name)
                        P.append(pid)
                else:
                    print("discarded!", folder_name, filename, len(data))

    X = np.asarray(X)
    Y = np.asarray(Y)
    P = np.asarray(P)

    # TODO: Anti-alliasing with 10 Hz

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

    print(f"Total samples: {total_samples}")
    print(f"Saved in {OUTDIR}")
    print("X shape:", X.shape)
    print("Y distribution:", len(set(Y)))
    print(pd.Series(Y).value_counts())
    print("User distribution:", len(set(P)))
    print(pd.Series(P).value_counts())


if __name__ == "__main__":
    main()
