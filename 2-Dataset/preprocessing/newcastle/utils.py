import numpy as np
from scipy.interpolate import interp1d
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import os
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri


SCRATCH = os.environ['SCRATCH']
SLEEP_LABELS = ["R", "W", "N1", "N2", "N3"]

def extract_number(s):
    number = int(re.findall(r'merged_participant_(\d+)', s)[0])
    return number

def label_sleep(x):
                if x in SLEEP_LABELS:
                    return 'sleep'
                else:
                    return np.NaN
def read_geneactiv_bin(filename):
    print(filename)
    robjects.r(f'''
    library(GENEAread)
    accdata <- read.bin("{filename}")
    str(accdata)
    xyz_data <- accdata$data.out[, c("timestamp", "x", "y", "z")]
    df <- as.data.frame(xyz_data)
    ''')

    # Get the data frame from R's global environment
    data = robjects.r['df']

    # Convert R data.frame to pandas DataFrame
    df = pandas2ri.rpy2py_dataframe(data)

    
    return df

def process_files():
    # Specify the directory where your .bin files are stored.
    directory = f'{SCRATCH}/mp-dataset/newcastlesleep/dataset_psgnewcastle2015_v1.0/acc/'

    # Get a list of all .bin files in the directory.
    files = [f for f in os.listdir(directory) if f.endswith('.bin')]

    # Group the files by participant number.
    files_by_participant = {}
    for file in files:
        # Extract the participant number from the file name.
        participant = int(file.split('_')[0].replace('MECSLEEP', ''))
        
        existing_file = directory + f"participant_{participant}.npy"
        if os.path.exists(existing_file):
            continue
        else:
            # Add the file to the list for this participant.
            if participant not in files_by_participant:
                files_by_participant[participant] = []
            files_by_participant[participant].append(file)
    print("Files appended")

    # For each participant, read the left and right wrist files and combine them.
    for participant, files in files_by_participant.items():
        print("Start bin files processing")
        if len(files) != 2:  # Ensure there are exactly 2 files for each participant.
            combined_df = df = read_geneactiv_bin(os.path.join(directory, file))

        dfs = []
        for file in files:
            # Read the file into a DataFrame.
            df = read_geneactiv_bin(os.path.join(directory, file))
            # Append the DataFrame to the list for this participant.
            dfs.append(df)

        # Combine the DataFrames for this participant.
        combined_df = pd.concat(dfs)

        # Save the combined DataFrame to a .npy file.
        print("Saving combined file")
        np.save(directory + f'participant_{participant}.npy', combined_df.to_numpy())


def labeling():
    participants = [1,2,10,14,17,21,23,27,28,29,31,32,34,35,38,39,42,45,48,49,50,51,52,53,56,57,59,60]
    print('mapping')
    mapping = {f"participant_{i}.npy": f"mecsleep{i:02}_psg.txt" for i in participants}
    print(mapping)

    # the directories of the npy and txt files
    npy_dir = f"{SCRATCH}/mp-dataset/newcastlesleep/dataset_psgnewcastle2015_v1.0/acc"
    txt_dir = f"{SCRATCH}/mp-dataset/newcastlesleep/dataset_psgnewcastle2015_v1.0/psg"
    # The directory where you want to save your merged npy files
    merged_npy_dir = npy_dir + "/merged"
    os.makedirs(merged_npy_dir, exist_ok=True)
    for npy_file, txt_file in mapping.items():
        # csv_file = f"{npy_dir}/merged_{npy_file[:-4]}.csv"
        
        merged_npy_file = f"{merged_npy_dir}/merged_{npy_file}"

        # Check if the output file already exists
        if os.path.exists(merged_npy_file):
            continue
        print("Creation of", merged_npy_file)
        # load the npy file
        data = np.load(f"{npy_dir}/{npy_file}")
        df = pd.DataFrame(data)
        
        # convert the first column to datetime
        df[0] = pd.to_datetime(df[0], unit='s').dt.floor('S')
        # Set the first column as the index
        df.set_index(0, inplace=True)
        
        # load the txt file
        df_sleep = pd.read_csv(f"{txt_dir}/{txt_file}", sep="\t", skiprows = 17)
        
        # convert the "Time [hh:mm:ss]" column to datetime
        df_sleep['Datetime'] = pd.to_datetime(df_sleep['Time [hh:mm:ss]'], format='%H:%M:%S').dt.time
        
        # initialize a date object
        date = df.index[0].date()
        
        # create a date series starting from the first date and incrementing by 1 day when the time passes midnight
        dates = []
        prev_time = df_sleep['Datetime'].iloc[0]  # initialize with the first time in your data
        for current_time in df_sleep['Datetime']:
            if current_time < prev_time:
                date += timedelta(days=1)
            dates.append(date)
            prev_time = current_time

        # replace df_sleep 'Datetime' column with new datetime values
        df_sleep['Datetime'] = [datetime.combine(d, t) for d, t in zip(dates, df_sleep['Datetime'])]

        # sort the df dataframe
        df = df.sort_index()

       # Sort both dataframes by their time columns
        df = df.sort_index()
        df_sleep = df_sleep.sort_values(by='Datetime')

        # Merge the dataframes using merge_asof with a 30-second tolerance
        merged_df = pd.merge_asof(df, df_sleep, left_index=True, right_on='Datetime', tolerance=pd.Timedelta('30s'))

                # Save the merged dataframe as a npy file
        print("Saving")
        np.save(merged_npy_file, merged_df.values)



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
             "sleep" : "sleep",
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