#!/bin/bash

export PATH2SOURCEROOT="$HOME/mp_project/t-st-2023-OneShot-HAR-Hyeongkyun-Orestis/3-Model/_inference"

# =============================================
# EVALUATION DATASET
export INF_DATA="unseen_W18"
export INF_DATASET_NAME="adl_30hz_w10"
export PATH2INF_DATA="$SCRATCH"/mp-dataset/__Important__/eval_$INF_DATASET_NAME/relabel/Willetts2018/
export USE_UNKNOWN="True"

# =============================================
# SETTING 1
# Choose finetuned model's weight used for inference.
export FTMODEL="3_xi_138e-2"
export PATH2FT_WEIGHT="$HOME/mp_project/t-st-2023-OneShot-HAR-Hyeongkyun-Orestis/3-Model/_finetuning/_3-all/_model/$FTMODEL"
# =============================================

# =============================================
# SETTING 2
export OUTPUTPREFIX="adl"
export SAVE_LOGITS="True"
export SAVE_FINAL_FEATS="True"
# =============================================

# =============================================
# SETTING 3 - For Method1
# Choosing Variants
export ADD_CLASSICAL_FEATS="True"
# =============================================



job_name="I-$INF_DATASET_NAME"

sbatch  <<EOT
#!/bin/bash
###############################
# Define sbatch configuration #
###############################

#SBATCH -A es_chatzi
#SBATCH -n 8
#SBATCH --time=5:00:00
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

python "${PATH2SOURCEROOT}/_inference.py" \
        model_root="$PATH2FT_WEIGHT" \
        output_prefix="$OUTPUTPREFIX" \
        gpu=1 \
        data=$INF_DATA \
        save_logits=$SAVE_LOGITS \
        save_final_feats=$SAVE_FINAL_FEATS \
        add_classical_feats=$ADD_CLASSICAL_FEATS\
        data.data_root="$PATH2INF_DATA" \
        data.dataset_name="$INF_DATASET_NAME" \
        data.inf_use_unknown=$USE_UNKNOWN \

EOT