script="./get_test_pids.py"
job_name="tmp" 

sbatch <<EOT
#!/bin/bash
###############################
# Define sbatch configuration #
###############################

#SBATCH -A es_chatzi
#SBATCH -n 8
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=10G
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

python $script
EOT