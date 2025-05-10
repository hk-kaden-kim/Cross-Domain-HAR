# 1-Environment Setup

This section details the setup process for the project environment, both on the ETH Zurich Euler cluster and locally. The setup ensures all necessary dependencies and the correct Python environment are in place for running the experiments.

## Initial One-Time Setup on the Euler Cluster

For those using the ETH Zurich Euler cluster, follow these initial setup steps. This setup is a one-time requirement to prepare your cluster environment for the experiments.

### Access and Documentation

1. Familiarize yourself with the Euler cluster documentation and usage rules:
   - Access the cluster following the [Accessing the cluster guide](https://scicomp.ethz.ch/wiki/Accessing_the_cluster).
   - Consult the [Getting started with clusters tutorial](https://scicomp.ethz.ch/wiki/Getting_started_with_clusters), especially sections 2, 3, and 5.
   - Review the [ETH Zurich Acceptable Use Policy for ICT ("BOT")](https://rechtssammlung.sp.ethz.ch/Dokumente/203.21en.pdf).

### Miniconda Installation

2. Connect to the cluster and ensure you're in your home directory (`cd ~`).
3. Install Miniconda by following these steps:
   1. Download Miniconda: `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`.
   2. Make the installer executable: `chmod +x Miniconda3-latest-Linux-x86_64.sh`.
   3. Run the installer: `./Miniconda3-latest-Linux-x86_64.sh`, then follow the on-screen instructions. Ensure installation in `/cluster/home/USERNAME/miniconda3`.
   4. When prompted to initialize Miniconda3, respond with `yes`.
   5. Disconnect and reconnect to the cluster to refresh your environment.
   6. Optional: Run `conda config --set auto_activate_base false` to prevent the base environment from auto-activating.
   7. Clean up by removing the installer script: `rm Miniconda3-latest-Linux-x86_64.sh`.

Note: This Miniconda installation step is specific to users on the Euler cluster and may not be necessary for local setups.

## Setting Up the MTSSL Conda Environment

The MTSSL project requires a specific Python environment, which can be set up using the provided `setup_mtssl_env.sh` script. This script automates the creation of the conda environment and installation of required packages.

### Automated Environment Setup

Run the following script located in the `1-Environment` folder to automatically set up the MTSSL conda environment:

```bash
./setup_mtssl_env.sh

