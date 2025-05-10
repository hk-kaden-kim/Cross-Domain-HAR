#!/bin/bash
###############################
# Define bash script #
###############################
echo "Do you want to change the labels to capture24?"
echo "--------------------"
echo "If you choose to 'n(no)', it will keep its original labels..."
echo " "
echo "original labels..."
echo "Keyboard typing", "Using mouse", "Handwriting", "Cutting vegetables", "Stir-frying vegetables",
echo "Wiping the table", "Sweeping floor", "Using vacuum to vacuum floor", "Open and close drawer", "None Activity"
echo "--------------------"
read -p "Enter your choice (y/n): " choice

# Store the absolute path to the script in a variable

ABSOLUTE_PATH="$HOME/mp_project/t-st-2023-OneShot-HAR-Hyeongkyun-Orestis"
script="$ABSOLUTE_PATH/2-Dataset/preprocessing/householdhu/mtssl_finetune_householdhu.py"

case $choice in 
    [yY] )
        job_name="ftdat-householdhu-relabel"
        relabel="--relabel"
        ;;
    [nN] )
        job_name="ftdat-householdhu-adl"
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

#SBATCH -n 4
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=16G
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