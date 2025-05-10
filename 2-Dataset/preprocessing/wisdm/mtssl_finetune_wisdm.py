"""
Create MTSSL finetuning data from the wisdm dataset.

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

DEVICE_HZ = 20  # Hz
WINDOW_SEC = 10  # seconds
WINDOW_OVERLAP_SEC = 5  # seconds
WINDOW_LEN = int(DEVICE_HZ * WINDOW_SEC)  # device ticks
WINDOW_OVERLAP_LEN = int(DEVICE_HZ * WINDOW_OVERLAP_SEC)  # device ticks
WINDOW_STEP_LEN = WINDOW_LEN - WINDOW_OVERLAP_LEN  # device ticks
WINDOW_TOL = 0.01  # 1%
TARGET_HZ = 30  # Hz
TARGET_WINDOW_LEN = int(TARGET_HZ * WINDOW_SEC)

SCRATCH = os.environ['SCRATCH']
DATAFILES = SCRATCH + "/mp-dataset/wisdm/wisdm-dataset/raw/watch/accel/*.txt"

OUTDIR = SCRATCH + f"/mp-dataset/finetune_wisdm_{TARGET_HZ}hz_w{WINDOW_SEC}/"
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

label_dict = {}
label_dict["walking"] = "A"
label_dict["jogging"] = "B"
label_dict["stairs"] = "C"
label_dict["sitting"] = "D"
label_dict["standing"] = "E"
label_dict["typing"] = "F"
label_dict["teeth"] = "G"
label_dict["soup"] = "H"
label_dict["chips"] = "I"
label_dict["pasta"] = "J"
label_dict["drinking"] = "K"
label_dict["sandwich"] = "L"
label_dict["kicking"] = "M"
label_dict["catch"] = "O"
label_dict["dribbling"] = "P"
label_dict["writing"] = "Q"
label_dict["clapping"] = "R"
label_dict["folding"] = "S"
code2name = {code: name for name, code in label_dict.items()}

FILTERED_LABEL = ["clapping"]
##########################################################################

def tmp(my_x):
    return float(my_x.strip(";"))

def main():

    X, Y, T, P, = ([],[],[],[],)
    
    column_names = ["pid", "code", "time", "x", "y", "z"]

    for datafile in tqdm(glob.glob(DATAFILES)):
        columns = ["pid", "class_code", "time", "x", "y", "z"]
        one_person_data = pd.read_csv(
            datafile,
            sep=",",
            header=None,
            converters={5: tmp},
            names=column_names,
            parse_dates=["time"],
            index_col="time",
        )
        one_person_data.index = pd.to_datetime(one_person_data.index)
        period = int(round((1 / DEVICE_HZ) * 1000_000_000))
        # one_person_data.resample(f'{period}N', origin='start').nearest(limit=1)
        code_to_df = dict(tuple(one_person_data.groupby("code")))
        pid = int(one_person_data["pid"][0])

        for code, data in code_to_df.items():
            try:
                data = data.resample(f"{period}N", origin="start").nearest(limit=1)
            except ValueError:
                if pid == 1629:
                    data = data.drop_duplicates()
                    data = data.resample(f"{period}N", origin="start").nearest(
                        limit=1
                    )
                    pass

            for i in range(0, len(data), WINDOW_STEP_LEN):
                w = data.iloc[i : i + WINDOW_LEN]

                if not is_good_quality(w, WINDOW_LEN, WINDOW_SEC, WINDOW_TOL):
                    continue

                x = w[["x", "y", "z"]].values
                t = w.index[0].to_datetime64()

                X.append(x)
                Y.append(code2name[code])
                T.append(t)
                P.append(pid)

    X = np.asarray(X)
    Y = np.asarray(Y)
    T = np.asarray(T)
    P = np.asarray(P)

    # fixing unit to g
    X = X / 9.81

    # downsample to 30 Hz
    X = resize(X, TARGET_WINDOW_LEN)
    # map wisdm labels to capture24 type
    if RELABEL:
        Y, idx_filter = label_mapto_capture24(Y,FILTERED_LABEL,is_Walmsley2020,is_Willetts2018)
        X = X[idx_filter,:,:]
        Y = Y[idx_filter]
        P = P[idx_filter]

    os.system(f"mkdir -p {OUTDIR}")
    np.save(os.path.join(OUTDIR, "X"), X)
    np.save(os.path.join(OUTDIR, "Y"), Y)
    np.save(os.path.join(OUTDIR, "time"), T)
    np.save(os.path.join(OUTDIR, "pid"), P)

    print(f"Saved in {OUTDIR}")
    print("X shape:", X.shape)
    print("Y distribution:", len(set(Y)))
    print(pd.Series(Y).value_counts())
    print("User distribution:", len(set(P)))
    print(pd.Series(P).value_counts())

if __name__ == "__main__":
    main()
