import numpy as np
import pandas as pd
import re, os
from scipy.interpolate import interp1d
from scipy import stats as s

def is_good_quality(w, WINDOW_LEN, WINDOW_SEC, WINDOW_TOL):
    """Window quality check"""

    # Check null values in selected windows
    if w.isna().any().any():
        return False

    # Check the selected window length
    if len(w) != WINDOW_LEN:
        return False
    
    w_start, w_end = w.index[0], w.index[-1]
    w_duration = w_end - w_start
    target_duration = pd.Timedelta(WINDOW_SEC, "s")
    if np.abs(w_duration - target_duration) > WINDOW_TOL * target_duration:
        return False
    return True
    
def resize(X, length, axis=1):
    """Resize the temporal length using linear interpolation.
    X must be of shape (N,M,C) (channels last) or (N,C,M) (channels first),
    where N is the batch size, M is the temporal length, and C is the number
    of channels.
    If X is channels-last, use axis=1 (default).
    If X is channels-first, use axis=2.
    """
    length_orig = X.shape[axis]
    t_orig = np.linspace(0, 1, length_orig, endpoint=True)
    t_new = np.linspace(0, 1, length, endpoint=True)
    X = interp1d(t_orig, X, kind="linear", axis=axis, assume_sorted=True)(
        t_new
    )
    return X

def parse_filename(name):
    WEAR_CODE = {"035": "waist", "064": "left wrist", "066": "right wrist"}
    pattern = re.compile(r"(P\d{2})_(\d{3})", flags=re.IGNORECASE)  # P01_35
    m = pattern.search(os.path.basename(name))
    part, wear = m.group(1), m.group(2)
    wear = WEAR_CODE[wear]
    return part, wear

def fix_labels(y):
    # Combine these labels as they're extremely rare
        if (
            (y == "Free-Living 10km/hr Run")
            or (y == "10km/hr Run")
            or (y == "12km/hr Run")
        ):
            y = "10+km/hr Run"
        # Combine these too
        if y == "Free-Living 6km/hr Walk":
            y = "6km/hr Walk"
        return y





def label_mapto_capture24(Y, FILTERED_LABEL, walmsley2020=False):
    # String data type change to U24
    Y = Y.astype('U24')
    
    mapping =   {
                    "lie":"sleep",
                    "sit":"sitting",
                    "stand":"standing",
                    "walk":"walking",
                } 

    if walmsley2020:
        print("Relabeling to Capture 24 label:Walmsley2020")
        for k in mapping.keys():
            willettsSpecific2018 = mapping[k]
            if willettsSpecific2018 in ["sleep"]:
                mapping[k] = "sleep"
            if willettsSpecific2018 in ["sitting","vehicle"]:
                mapping[k] = "sedentary"
            if willettsSpecific2018 in ["standing", "walking", "household-chores"]:
                mapping[k] = "light"
            if willettsSpecific2018 in ["sports","bicycling"]:
                mapping[k] = "Moderate-vigorous"
    else:
        print("Relabeling to Capture 24 label:WillettsSpecific2018")

    idx_filter = []
    for idx, label in enumerate(Y):
        if label not in FILTERED_LABEL:
            Y[idx] = mapping[label]
            idx_filter.append(True)
        else:
            Y[idx] = "None"
            idx_filter.append(False)
    if not all(idx_filter):
        print(f"\t data with the label {FILTERED_LABEL} are filtered out. {idx_filter.count(False)}")
    
    return Y, idx_filter