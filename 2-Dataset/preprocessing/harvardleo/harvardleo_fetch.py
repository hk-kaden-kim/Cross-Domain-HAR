from pyDataverse.api import NativeApi,DataAccessApi
from pyDataverse.models import Dataverse
import pandas as pd
import tqdm
import os


# get the digital object identifier for the Dataverse dataset
DOI = "doi:10.7910/DVN/G23QTS"


# establish connection
base_url = 'https://dataverse.harvard.edu/'
api = NativeApi(base_url)
print(api)
# print(api.status)

# retrieve the contents of the dataset
data_api = DataAccessApi(base_url)

dataset = api.get_dataset(DOI)
# Get a list of all .tab files that have 'wrist' in their name
files_list = dataset.json()['data']['latestVersion']['files']
# Filter the list to keep only files with 'wrist' in the filename
wrist_files = [file for file in files_list if 'wrist' in file['dataFile']['filename']]

# Set the output folder path
output_folder = os.getenv("SCRATCH") + "/mp-dataset/harvardleo/"

# Create the output folder if it does not exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
print("Start Downloading")
for file in tqdm.tqdm(wrist_files, unit='file', ncols=os.get_terminal_size().columns, position=0):
    filename = file["dataFile"]["filename"]
    if 'wrist' in file['dataFile']['filename']: 
        file_id = file["dataFile"]["id"]
        print("File name {}, id {}".format(filename, file_id), end='\r')
        response = data_api.get_datafile(file_id)
        with open(os.path.join(output_folder, filename), "wb") as f:
            f.write(response.content)
    else:
        continue
print("Download successful")