#!/bin/bash
###############################
# Define bash script #
###############################
echo "Do you want to change the labels to capture24?"
echo "--------------------"
echo "If you choose to 'n(no)', it will keep its original labels..."
echo " "
echo "original labels..."
echo "brush teeth", "computer", "dinner", "shower", "shopping", "commuting", "exercise", "lunch", "work"
echo "--------------------"
read -p "Enter your choice (y/n): " choice

# Store the absolute path to the script in a variable

ABSOLUTE_PATH="$HOME/mp_project/t-st-2023-OneShot-HAR-Hyeongkyun-Orestis"
script="$ABSOLUTE_PATH/2-Dataset/preprocessing/commuting/mtssl_finetune_commuting.py"

case $choice in 
    [yY] )
        job_name="ftdat-commuting-relabel"
        relabel="--relabel"
        ;;
    [nN] )
        job_name="ftdat-commuting-adl"
        relabel=""
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
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=5G
#SBATCH --job-name=$job_name

echo "###############################################################"
echo ""
echo "Slurm JOB Logs"
echo "..."
echo ""
echo ""
echo "###############################################################"

python $script $relabel
EOT