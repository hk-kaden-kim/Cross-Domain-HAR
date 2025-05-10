#!/bin/bash
###############################
# Define bash script #
###############################
echo "Do you want to change the labels to capture24?"
echo "--------------------"
echo "If you choose to 'n(no)', it will keep its original labels..."
echo " "
echo "original labels..."
echo "writing","type_on_a_keyboard","washing_dishes","ironing","washing_hands","brush_teeth","brush_hair",
echo "dusting","put_on_a_shoe","put_on_a_jacket","take_off_a_jacket","take_off_a_shoe","drink_water","blow_nose",
echo "phone_call","eat_meal","put_on_glasses","sit_down",
echo "open_a_bottle","salute","sneeze_cough","stand_up","open_a_box","take_off_glasses"
echo "--------------------"
read -p "Enter your choice (y/n): " choice

# Store the absolute path to the script in a variable

ABSOLUTE_PATH="$HOME/mp_project/t-st-2023-OneShot-HAR-Hyeongkyun-Orestis"
script="$ABSOLUTE_PATH/2-Dataset/preprocessing/paal/mtssl_finetune_paal.py"

case $choice in 
    [yY] )
        job_name="ftdat-paal-relabel"
        relabel="--relabel"
        ;;
    [nN] )
        job_name="ftdat-paal-adl"
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