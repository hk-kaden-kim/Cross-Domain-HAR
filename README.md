# Cross-Domain HAR: Self-supervised Learning and Enhanced Fine-tuning Approaches

## Overview
This project addresses the challenge of generalizing Human Activity Recognition (HAR) models across varied datasets from wrist-worn accelerometers. We explore Multi-Task Self-Supervised Learning (MTSSL) and advanced fine-tuning techniques, including classic feature extraction, unknown sample integration, specialized loss functions, and diverse data augmentation strategies. Our approach markedly improves the cross-domain generalization and adaptability of HAR models, demonstrated through significant accuracy and performance enhancements in classification tasks. 

### Citation
```
Kim, H., Oikonomou, O. (2024). Cross-Domain HAR: Self-supervised Learning and Enhanced Finetuning Approaches. Master's Project, University of Zürich AIML Group, and ETH Zürich Sensing, Interaction & Perception Group.
https://seafile.ifi.uzh.ch/f/5c79a0dfef5d4293a5c7/
```


## Project Structure
This repository is organized into four main folders, each dedicated to a specific aspect of the project's development and execution. Below is an outline of each folder and its contents:

### 1. Environment
**Description**: Contains files and scripts for setting up the project environment, including dependencies and libraries. You can find more details in the readme file [here](./1-Environment/README.md).

**Contents**:
- Dependency management files (`requirements.txt`, `environment.yml`)
- Bash scripts for environment setup

### 2. Dataset
**Description**: Includes scripts for fetching, preprocessing, and managing datasets for model training and testing. You can find more details in the readme file [here](./2-Dataset/README.md).

**Contents**:
- Dataset acquisition scripts
- Data preprocessing scripts

### 3. Model
**Description**: Contains training, inference, and configuration files for various model variations, focusing on enhancing model performance. You can find more details in the readme file [here](./3-Model/README.md).

**Contents**:
- Training and inference scripts
- Configuration files for model variations
- Bash scripts for model operations

### 4. Evaluation
**Description**: Dedicated to evaluating model performance through various metrics and analyses. You can find more details in the readme file [here](./4-Evaluation/README.md).

**Contents**:
- Scripts for performance evaluation
- Tools for semantic analysis and misclassification trends

## Getting Started
To begin with this project, clone the repository and follow the setup instructions in the `Environment` folder. Prepare your datasets as described in the `Dataset` folder, then proceed to model training and inference with guidance from the `Model` folder. Evaluate the models' performance through scripts found in the `Evaluation` folder.

## Contribution
We welcome contributions to improve the project. If interested, please refer to the contribution guidelines or open an issue to discuss your ideas.
