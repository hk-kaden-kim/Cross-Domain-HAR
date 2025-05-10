# 2-Dataset Preparation

<b>NOTE!</b> Before processing the code here. Check the conda environment is activated properly.

## Overview

This document provides comprehensive guidance for fetching, preprocessing, and customizing the datasets required for this project. Designed to prepare datasets for both pre-training and fine-tuning processes, it includes steps for downloading data, preparing it according to specific requirements, and adjusting settings to suit various research needs.

Because human activity datasets have large storage, every dataset and preprocessing result will be saved into your personal $SCRATCH/mp-dataset folder. mp-dataset folder will be created automatically into the $SCRATCH folder.
```
echo $SCRATCH   # To check the absolute path of your $SCRATCH folder.
```
## Fetching Datasets

To download the required datasets, navigate to the `fetching` subfolder and execute the `script_fetch_data.sh`. This script automates the download, extraction, and initial organization of the datasets.

```bash
cd fetching
bash script_fetch_data.sh
```
Upon execution of the dataset fetching script, you will be prompted to choose the dataset you wish to download, with each option corresponding to a different dataset. The script manages the download, unzipping, and basic organization of the selected dataset.

## Dataset Preprocessing

Following download, datasets require uniform preprocessing to ensure compatibility with the pre-training and fine-tuning stages. Preprocessing scripts specific to each dataset are located within the `preprocessing/(Dataset name)` directory.

```bash
bash preprocessing/(Dataset name)/script_preprocess_*.sh
```

## Specific Instructions:

- **Capture 24 dataset**: You'll choose between preprocessing for fine-tuning or pre-training. It's important to note that preprocessing for the pre-training dataset should be done after processing the fine-tuning dataset in our code.

- **Other datasets**: The script will inquire whether you wish to retain the original labeling or switch to the Capture24 dataset labeling.

- **Negative and Unknown samples**: After fetching all datasets, you can generate negative and unknown samples for the source domain experiment. The code in *./preprocessing/create_source_unknown_set.py* will create several folders in your dataset root.
  -  ./(ROOT)/finetune_negative_30hz_w10 : Negative samples used for finetuning process.
  -  ./(ROOT)/test_negative_30hz_w10 : Negative samplese used for evaluation process.
  -  ./(ROOT)/test_unknown_30hz_w10 : Unknown samplese used for evaluation process.
  -  ./(ROOT)/test_unknown_all_30hz_w10 : All samples from 'test_negative_30hz_w10' and 'test_unknown_30hz_w10'


## Customizing Dataset Storage and Preprocessing Parameters

### Changing the Storage Location

To modify where datasets are stored, adjust the `OUTPUT_ROOT` variable in `script_fetch_data.sh`:

```bash
OUTPUT_ROOT="$SCRATCH/mp-dataset"
```
Additionally, make sure the paths in the preprocessing scripts within the `preprocessing/(Dataset name)` directories are updated as needed.

## Adjusting Preprocessing Specifications

Within each dataset's preprocessing script (e.g., `mtssl_finetune_(dataset_name).py`), parameters such as `DEVICE_HZ`, `WINDOW_SEC`, and other relevant settings can be customized to fit the requirements of your project.

## Relabeling Frameworks

This project employs three different frameworks for dataset relabeling, based on specific study requirements:

- `is_willets2018`: [Reference](https://doi.org/10.1038/s41598-018-26174-1)
- `is_Walmsley2020`: [Reference](https://doi.org/10.1101/2020.11.10.20227769)
- When neither `is_Walmsley2020` nor `is_willets2018` is true, relabeling adheres to the framework used in the original study of each dataset

Ensure these settings in the preprocessing scripts are adjusted to align dataset labels with the selected framework.

