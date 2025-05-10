import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy import stats as s

def is_good_quality(w, WINDOW_LEN):
    """Window quality check"""

    # Check null values in selected windows
    if w.isna().any().any():
        print('nan')
        return False

    # Check the selected window length
    if len(w) != WINDOW_LEN:
        print('len')
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



def content2x_and_y(data_content, epoch_len=30, sample_rate=100, overlap=15):
    sample_count = int(np.floor(len(data_content) / (epoch_len * sample_rate)))

    sample_label_idx = 1
    sample_x_idx = 2
    sample_y_idx = 3
    sample_z_idx = 4

    sample_limit = sample_count * epoch_len * sample_rate
    data_content = data_content[:sample_limit, :]

    label = data_content[:, sample_label_idx]
    x = data_content[:, sample_x_idx]
    y = data_content[:, sample_y_idx]
    z = data_content[:, sample_z_idx]

    # to make overlappting window
    offset = overlap * sample_rate
    shifted_label = data_content[offset:-offset, sample_label_idx]
    shifted_x = data_content[offset:-offset:, sample_x_idx]
    shifted_y = data_content[offset:-offset:, sample_y_idx]
    shifted_z = data_content[offset:-offset:, sample_z_idx]

    shifted_label = shifted_label.reshape(-1, epoch_len * sample_rate)
    shifted_x = shifted_x.reshape(-1, epoch_len * sample_rate, 1)
    shifted_y = shifted_y.reshape(-1, epoch_len * sample_rate, 1)
    shifted_z = shifted_z.reshape(-1, epoch_len * sample_rate, 1)
    shifted_X = np.concatenate([shifted_x, shifted_y, shifted_z], axis=2)

    label = label.reshape(-1, epoch_len * sample_rate)
    x = x.reshape(-1, epoch_len * sample_rate, 1)
    y = y.reshape(-1, epoch_len * sample_rate, 1)
    z = z.reshape(-1, epoch_len * sample_rate, 1)
    X = np.concatenate([x, y, z], axis=2)

    X = np.concatenate([X, shifted_X])
    label = np.concatenate([label, shifted_label])
    return X, label

def clean_up_label(X, labels, LABEL_NAMES):
    # 1. remove rows with >50% zeros
    sample_count_per_row = labels.shape[1]

    rows2keep = np.ones(labels.shape[0], dtype=bool)
    transition_class = 0
    for i in range(labels.shape[0]):
        row = labels[i, :]
        if np.sum(row == transition_class) > 0.5 * sample_count_per_row:
            rows2keep[i] = False

    labels = labels[rows2keep]
    X = X[rows2keep]

    # 2. majority voting for label in each epoch
    final_labels = []
    for i in range(labels.shape[0]):
        row = labels[i, :]
        mode = int(s.mode(row)[0][0])
        final_labels.append(LABEL_NAMES[mode])
    final_labels = np.array(final_labels, dtype='U24')
    # print("Clean X shape: ", X.shape)
    # print("Clean y shape: ", final_labels.shape)
    return X, final_labels



def label_mapto_capture24(Y, FILTERED_LABEL, walmsley2020=False, willetts2018=False):
    # String data type change to U24
    Y = Y.astype('U24')
    
    mapping =   {
                    "lying":"sleep",
                    "sitting":"sitting",
                    "standing":"standing",
                    "walking":"walking",
                    "running":"sports",
                    "cycling":"bicycling",
                    "Nordic walking":"walking", #change
                    "watching TV":"sitting", #change
                    "computer work":"sitting",
                    "car driving":"vehicle",
                    "ascending stairs":"walking",
                    "descending stairs":"walking",
                    "vacuum cleaning":"unknown", #change
                    "ironing":"household-chores",
                    "folding laundry":"household-chores",
                    "house cleaning":"unknown", #change
                    "playing soccer":"unknown", #change
                    "rope jumping":"unknown", #change
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