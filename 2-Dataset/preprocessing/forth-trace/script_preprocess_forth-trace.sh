#!/bin/bash
###############################
# Define bash script #
###############################
echo "Do you want to change the labels to capture24?"
echo "--------------------"
echo "If you choose to 'n(no)', it will keep its original labels..."
echo " "
echo "original labels..."
echo '"stand","sit","sit and talk","walk","walk and talk","climb stairs (up/down)","climb stairs (up/down) and talk","stand -> sit","sit -> stand",'
echo '"stand -> sit and talk","sit and talk -> stand","stand -> walk","walk -> stand","stand -> climb stairs (up/down), stand -> climb stairs (up/down) and talk",'
echo '"climb stairs (up/down) -> walk","climb stairs (up/down) and talk -> walk and talk"'
echo "--------------------"
read -p "Enter your choice (y/n): " choice

# Store the absolute path to the script in a variable

ABSOLUTE_PATH="$HOME/mp_project/t-st-2023-OneShot-HAR-Hyeongkyun-Orestis"
script="$ABSOLUTE_PATH/2-Dataset/preprocessing/forth-trace/mtssl_finetune_forth-trace.py"

case $choice in 
    [yY] )
        job_name="ftdat-forth-trace-relabel"
        relabel="--relabel"
        ;;
    [nN] )
        job_name="ftdat-forth-trace"
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

#SBATCH -n 1
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=1G
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