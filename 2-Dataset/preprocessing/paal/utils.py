import numpy as np
from scipy.interpolate import interp1d

def is_good_quality(w, WINDOW_LEN):
    """Window quality check"""

    # Check null values in selected windows
    if w.isna().any().any():
        return False

    # Check the selected window length
    if len(w) != WINDOW_LEN:
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
    Y = Y.astype('U24')
    
    # Default mapping is corresponding to WillettsSpecific2018
    mapping =   {
             "writing" : "sitting",
             "type_on_a_keyboard" : "sitting",
             "washing_dishes" : "household-chores",
             "ironing" : "household-chores",
             "washing_hands" : "household-chores",
             "brush_teeth" : "standing",
             "brush_hair" : "standing",
             "dusting" : "household-chores",
             "put_on_a_shoe" : "unknown", #change 
             "put_on_a_jacket" : "unknown", #change 
             "take_off_a_jacket" : "unknown", #change 
             "take_off_a_shoe" : "unknown", #change 
             "drink_water" : "sitting",
             "blow_nose" : "unknown", #change 
             "phone_call" : "sitting",
             "eat_meal" : "sitting",
             "put_on_glasses" : "unknown", #change 
             "sit_down" : "sitting",
             "open_a_bottle" : "unknown", #change 
             "salute" : "unknown", #change 
             "sneeze_cough" : "unknown", #change 
             "stand_up" : "standing",
             "open_a_box" : "mixed-activity", #change 
             "take_off_glasses" : "unknown", #change 
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