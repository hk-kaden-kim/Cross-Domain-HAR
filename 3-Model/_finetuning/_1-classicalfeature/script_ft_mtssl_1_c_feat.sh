#!/bin/bash

export PATH2SOURCEROOT="$HOME/mp_project/t-st-2023-OneShot-HAR-Hyeongkyun-Orestis/3-Model/_finetuning"
export PATH2REPORT="$SCRIPT_PATH"/_1-classicalfeature/_model/

# =============================================
# SETTING 0
export OUTPUTPREFIX="1_c_feat" # Output model name : (OUTPUTPREFIX)_(cfg.data.dataset_name)

# =============================================
# SETTING 1
# Choose finetuning dataset
# e.g.
#       FT_DATA         PATH2FT_DATA    
#       cap24_W18       .../mp-dataset/finetune_capture24_30hz_w10/Willetts2018/                   
export FT_DATA="cap24_W18"

export PATH2FT_DATA="$SCRATCH"/mp-dataset/finetune_capture24_30hz_w10/Willetts2018/

# =============================================
# SETTING 2
# Pretraining Output Model Path
# e.g.
#       Pretrained with...              PATH2PT_WEIGHT
#       UKBB                            .../3-Model/_pretraining/_model/mtl_best.mdl
#       Capture24                       .../3-Model/_pretraining/_model/ssl_capture24.mdl
export PATH2PT_WEIGHT="$HOME/mp_project/t-st-2023-OneShot-HAR-Hyeongkyun-Orestis/3-Model/_pretraining/_model/mtl_best.mdl"
# =============================================

# =============================================
# SETTING 3
# Choosing Augmentation Techniques
# e.g.
#       None                    0
#       Axis Switch             1
#       Axis Rotate             2
#       Amplitued Scaling       3
export AUGMENTATION=0
# =============================================

# =============================================
# SETTING 4
# Choosing Variants
export ADD_CLASSICAL_FEATS="True"
# =============================================


job_name="$OUTPUTPREFIX"

sbatch  <<EOT
#!/bin/bash
###############################
# Define sbatch configuration #
###############################

#SBATCH -A es_chatzi
#SBATCH -n 8
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=5G
#SBATCH --job-name=$job_name
#SBATCH --gpus=1
#SBATCH --gres=gpumem:10G

echo "###############################################################"
echo ""
echo "Slurm JOB Logs"
echo "..."
echo ""
echo ""
echo "###############################################################"

python "${PATH2SOURCEROOT}/_finetuning_v2.py" \
        output_prefix="$OUTPUTPREFIX" \
        report_root="$PATH2REPORT" \
        gpu=1 \
        data=$FT_DATA \
        add_classical_feats=$ADD_CLASSICAL_FEATS\
        data.data_root="$PATH2FT_DATA" \
        evaluation.flip_net_path="$PATH2PT_WEIGHT" \
        data.augmentation=$AUGMENTATION \
        
EOT