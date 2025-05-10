from zipfile import ZipFile
import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

def is_good_quality(w, WINDOW_LEN):
    """Window quality check"""

    # Check null values in selected windows
    if np.isnan(w).any():
        # print('na')
        return False

    # Check the selected window length
    if len(w) != WINDOW_LEN:
        # print('len',len(w))
        return False
    
    # TODO : Do we need to check the label variety and window length tolerance,
    #       , like capture24 finetune code?

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

def label_mapto_capture24(Y, FILTERED_LABEL, walmsley2020=False, willetts2018=False):
    # String data type change to U24
    # Y = Y.astype('U24')
    print(Y)
    
    mapping =   {
                    "stand": 'standing',
                    "sit": 'sitting',
                    "sit and talk": 'sitting',
                    "walk": 'walking',
                    "walk and talk": 'walking',
                    "climb stairs (up/down)": 'walking',
                    "climb stairs (up/down) and talk": 'walking',
                    "stand -> sit": 'sitting', 
                    "sit -> stand": 'standing', 
                    "stand -> sit and talk": 'sitting', 
                    "sit and talk -> stand": 'standing', 
                    "stand -> walk": 'unknown', #change
                    "walk -> stand": 'unknown', #change
                    "stand -> climb stairs (up/down), stand -> climb stairs (up/down) and talk": 'unknown', #change
                    "climb stairs (up/down) -> walk": 'walking',
                    "climb stairs (up/down) and talk -> walk and talk": 'walking'
                                    } 


                
    if walmsley2020:
        print("Relabeling to Capture 24 label:Walmsley2020")
        for k in mapping.keys():
            willettsSpecific2018 = mapping[k]
            # if willettsSpecific2018 in ["sleep"]:
            #     mapping[k] = "sleep"
            if willettsSpecific2018 in ["sitting","vehicle"]:
                mapping[k] = "sedentary"
            elif willettsSpecific2018 in ["standing", "walking", "household-chores"]:
                mapping[k] = "light"
            elif willettsSpecific2018 in ["sports","bicycling"]:
                mapping[k] = "moderate-vigorous"
            else:
                True
    elif willetts2018:
        print("Relabeling to Capture 24 label:Willetts2018")
        for k in mapping.keys():
            willettsSpecific2018 = mapping[k]
            if willettsSpecific2018 in ["standing","sitting"]:
                mapping[k] = "sit-stand"
            elif willettsSpecific2018 in ["sports", "mixed-activity", "manual-work", "household-chores"]:
                mapping[k] = "mixed"
            else:
                True
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