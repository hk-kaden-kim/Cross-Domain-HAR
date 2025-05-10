#!/bin/bash
###############################
# Define bash script #
###############################
echo "Which script would you like to run?"
echo "1. Fine tuning"
echo "2. Pretraining"
echo "Attention if you havent performed the fine tuning then perform it before you choose the pretrain script"

read -p "Enter your choice (1 or 2): " choice

# Store the absolute path to the script in a variable

ABSOLUTE_PATH="$HOME/mp_project/t-st-2023-OneShot-HAR-Hyeongkyun-Orestis"

case $choice in 
    1)
        script="$ABSOLUTE_PATH/2-Dataset/preprocessing/capture24/mtssl_finetune_capture24.py"
        job_name="ftdat-capture24"
        
        ;;
    2)
        script="$ABSOLUTE_PATH/2-Dataset/preprocessing/capture24/mtssl_pretrain_capture24.py"
        job_name="ptdat-capture24"
        ;;
    *)
        echo "Invalid choice!"
        return 0
        ;;
esac
echo "Check your job with"
echo "$ myjobs -j <job id>"
echo "$ squeue "

sbatch <<EOT
#!/bin/bash
###############################
# Define sbatch configuration #
###############################

#SBATCH -n 2
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=50G
#SBATCH --job-name=$job_name

echo "###############################################################"
echo ""
echo "Slurm JOB Logs"
echo "..."
echo ""
echo ""
echo "###############################################################"

python $script
EOT