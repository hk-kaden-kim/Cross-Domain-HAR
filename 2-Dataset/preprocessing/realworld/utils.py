from zipfile import ZipFile
import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

def is_numpy_array_good_quality(window, WINDOW_LEN):
    """Window quality check"""

    if np.isnan(window).any():
        return False

    if len(window) != WINDOW_LEN:
        return False

    return True

def cal_min_time_len(readMe):
    time_len_list = []
    for line in readMe.readlines():
        if line.startswith(b"> entries"):
            time_len = int(line[11:16])
            time_len_list.append(time_len)

    min_time_len = min(time_len_list)
    print("min_time_len: ", min_time_len)
    return min_time_len

def cal_xyz(zip_file, csv, min_time_len):
    f = zip_file.open(csv)
    df = pd.read_csv(f)
    x = df["attr_x"]
    x = np.array(x)
    x = np.expand_dims(x, 1)
    y = df["attr_y"]
    y = np.array(y)
    y = np.expand_dims(y, 1)
    z = df["attr_z"]
    z = np.array(z)
    z = np.expand_dims(z, 1)
    xyz = np.concatenate((x, y, z), axis=1)
    xyz = xyz[:min_time_len, :]
    f.close()
    return xyz

def cal_xyz_acc(zip_file, i, sub_id, label, body_parts):
    readme_file = zip_file.open("readMe")
    min_time_len = cal_min_time_len(readme_file)

    xyz_acc = None
    for b_part in body_parts:
        if i == 1:
            con1 = sub_id == 4 and label == "walking"
            con2 = sub_id == 6 and label == "sitting"
            con3 = sub_id == 7 and label == "sitting"
            con4 = sub_id == 8 and label == "standing"
            con5 = sub_id == 13 and label == "walking"
            if con1 or con2 or con3 or con4 or con5:
                csv_file_name = "acc_{}_2_{}.csv".format(label, b_part)
            else:
                csv_file_name = "acc_{}_{}.csv".format(label, b_part)
        else:
            csv_file_name = "acc_{}_{}_{}.csv".format(label, i, b_part)
        if csv_file_name in zip_file.namelist():
            xyz = cal_xyz(zip_file, csv_file_name, min_time_len)
            if b_part == "chest":
                xyz_acc = xyz
            else:
                xyz_acc = np.concatenate((xyz_acc, xyz), axis=1)
        else:
            print(
                "data missing: {}/{}; missing part: {}".format(
                    sub_id, label, b_part
                )
            )
            xyz = np.full([min_time_len, 3], np.nan)
            if b_part == "chest":
                xyz_acc = xyz
            else:
                xyz_acc = np.concatenate((xyz_acc, xyz), axis=1)
    return xyz_acc

def intermediate_preprocess(dataset_path, output_path, pids, label_names, body_parts):
    print("Start the intermediate preprocess for realworld dataset")
    print("- Use only accerelometer dataset")
    print("- Create npy files for each participant and corresponding labels\n")
    # Check the output folder exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # For each participant
    for sub_id in pids:
        imu_data_path = dataset_path + "/proband{}/data/".format(sub_id)

        # For each labels
        for label in label_names:
            ziplabel = label.replace("_", "")
            # Use only accerelometer data. Filter out other sensor data.
            zip_file_path = imu_data_path + "acc_{}_csv.zip".format(ziplabel)
            z_file = ZipFile(zip_file_path)
            if z_file.namelist()[0].endswith("zip"):
                save_zip_path = imu_data_path + "acc_{}_csv/".format(ziplabel)
                z_file.extractall(path=save_zip_path)

                for i in range(1, len(z_file.namelist()) + 1):
                    sub_z_file = ZipFile(
                        save_zip_path + "acc_{}_{}_csv.zip".format(ziplabel, i)
                    )
                    xyz_acc = cal_xyz_acc(sub_z_file, i, sub_id, ziplabel, body_parts)
                    # save
                    file_name = "{}.{}.{}.npy".format(sub_id, i - 1, label)
                    np.save(os.path.join(output_path,file_name), xyz_acc)
                    sub_z_file.close()
                print(
                    "multiple: {}/{}; {} sessions in total".format(
                        sub_id, label, i
                    )
                )
            else:
                xyz_acc = cal_xyz_acc(z_file, 1, sub_id, ziplabel, body_parts)
                # save
                file_name = "{}.0.{}.npy".format(sub_id, label)
                np.save(os.path.join(output_path,file_name), xyz_acc)
                print("{}/{}".format(sub_id, label))

            z_file.close()

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
    
    mapping =   {
                    'climbingdown': 'walking', 
                    'climbingup': 'walking', 
                    'walking': 'walking', 
                    'standing': 'standing', 
                    'running': 'sports', 
                    'jumping': 'unknown', # changed
                    'lying': 'sleep', 
                    'sitting': 'sitting'
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