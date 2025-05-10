echo "-----------------------------------"
echo $"Start setup the environment ..."

echo -e "\n\n\n"
echo "1. Checking conda version ..."
conda --version

echo -e "\n\n\n"
echo "2. Create new conda python environment ..."
conda create -n mtssl_env python=3.7

echo -e "\n\n\n"
echo "3. Conda activate ..."
conda activate mtssl_env

echo -e "\n\n\n"
echo "4. Install related python packages ..."
pip install -r requirements.txt

echo -e "\n\n\n"
echo "DONE! - conda environment activated"
echo "-----------------------------------"
