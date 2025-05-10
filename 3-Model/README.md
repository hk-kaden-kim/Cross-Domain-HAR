# 3-Model Directory Overview

The `3-Model` directory is structured into several key subdirectories, each serving a specific role in the model development and deployment pipeline. Here's what each contains:

- `_conf`: Configuration files for fine-tuning and inference processes.
- `_finetuning`: Scripts and resources for fine-tuning models on specific datasets.
- `_inference`: Scripts for running inference with the fine-tuned models.
- `_pretraining/_model`: Stored weights from the pre-trained models.
- `sslearning`: Base fine-tuning code and architecture definitions from previous studies.

## SSL Learning

The `sslearning` subdirectory contains all the foundational code for fine-tuning, derived from prior research. Here you'll find basic network architectures and other necessary components for the SSL framework.

- **Base SSL Framework is based on this code**: [(Link)](https://github.com/OxWearables/ssl-wearables)

## Pretraining Models

Within `_pretraining/_model`, we store the weights from the pre-trained model, offering a jumpstart to the fine-tuning process based on previous research findings. In this project, we used the 'mtl_best.mdl' pre-trained weights [(Link)](https://github.com/OxWearables/ssl-wearables/tree/main/model_check_point).

## Fine-Tuning

The fine-tuning phase is subdivided into several categories, each within its own subdirectory under `_finetuning`. These are:

- `_1-classicalfeature`
- `_2-addunknown`
- `_3-all`
- `_4-augtech`
- `_baseline`

Each folder contains scripts named in the format `script_ft_mtssl_*.sh`, tailored for different aspects of fine-tuning, such as baseline adjustments, classical feature incorporation, unknown sample handling, and various augmentation techniques.

### Script Example

Here is a generalized structure of a fine-tuning script:

```bash
#!/bin/bash

# Define paths
export PATH2SOURCEROOT="path/to/_finetuning"
export PATH2REPORT="path/to/_baseline/_model/"

# Configuration settings
export OUTPUTPREFIX="base"
export FT_DATA="cap24_W18"
export PATH2FT_DATA="path/to/dataset"
export PATH2PT_WEIGHT="path/to/pretrained/model"

export AUGMENTATION=0

# Job submission
sbatch <<EOT
#!/bin/bash
# SBATCH configs

echo "Job Details"

# Call fine-tuning script
python "${PATH2SOURCEROOT}/_finetuning_v2.py" \
        --output_prefix="${OUTPUTPREFIX}" \
        --report_root="${PATH2REPORT}" \
        --gpu=1 \
        --data="${FT_DATA}" \
        --data_root="${PATH2FT_DATA}" \
        --pretrained_model_path="${PATH2PT_WEIGHT}" \
        --augmentation="${AUGMENTATION}"
EOT
```
- **Model Storage Path**: Adjust `PATH2SOURCEROOT` and `PATH2REPORT` to point to the desired storage locations.
- **Dataset Directory**: Modify `PATH2FT_DATA` to reflect the dataset's location.
- **Pretrained Model Weights**: Change `PATH2PT_WEIGHT` to the path of the pretrained model.
- **Configuration Files**: To alter fine-tuning settings, update the `config_ft.yaml` file within the `_conf` directory.

## Inference

The `_inference` directory follows a similar structure to `_finetuning`, with specific scripts designated for inference tasks.

- script_inf_mtssl_(model)_seen.sh: This bash script generates all inferences of the source domain samples by using the (model).
- script_inf_mtssl_(model)_unseen.sh: This bash script generates all inferences of the target domain samples by using the (model).

**NOTE!** in the case of model var-CUA(./_inference/_4-augtech), the argumentation technique you want to use should be set up manually through a variable below.
```
export FTMODEL="4_scl"
# 4_axsw        4_axr           4_scl
# 4_axsw_axr    4_axr_scl       4_axsw_scl
# 4_axsw_axr_scl      
```

