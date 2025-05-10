# 4-Evaluation

This directory contains scripts and notebooks used for the evaluation phase of our project. Each file is designed to assist in assessing the performance of our models in both open-set and closed-set scenarios, as well as analyzing the semantic meaning of deep feature spaces.

## Contents

- `Constants.py`: Contains constants used across multiple scripts for consistency.
- `README.md`: Provides an overview and guidance on how to use the scripts and notebooks in this directory.
- `evaluation.ipynb`: A Jupyter notebook that demonstrates how to perform evaluation tasks such as computing various metrics and visualizing results.
- `oscr_s_gpu.py`: Script for evaluating open-set classification rates using a GPU for speed improvements.
- `oscr_t_gpu.py`: Similar to `oscr_s_gpu.py` but tailored for target domain data in transfer learning scenarios.
- `utilAnalysis.py`: Utility functions for analysis including metrics computation, data sampling, and feature space reduction.

### utilAnalysis.py

This script includes functions for both open-set and closed-set evaluation, along with utilities for semantic analysis. Key functions include:

- Open-Set related:
  - `create_oscr_curve()`: Computes open-set classification rates.
  - `create_balanced_oscr_curve()`: Computes balanced open-set classification rates, taking into account class imbalance.
  - `create_balanced_oscr_curve_gpu()`: A GPU-accelerated version of the balanced OSCR computation.

- Closed-Set related:
  - `cal_bal_accuracy()`: Computes balanced accuracy for closed-set scenarios.
  - `multi_class_eval()`: Computes F1-score, Cohen's kappa, and balanced accuracy.
  - `get_confusionMatrix()`: Generates a confusion matrix for visual evaluation.

- Semantic Analysis:
  - `sampling_by_label()`: Samples the data based on labels to ensure balanced representation.
  - `reduceme()`: Reduces the dimensionality of the feature space for visualization purposes.
  - `reduceme_3D()`: Similar to `reduceme()` but reduces the feature space to three dimensions.

### evaluation.ipynb

Provides a step-by-step guide to evaluating model performance. It includes loading the data, computing evaluation metrics, and visualizing the outcomes. The notebook is designed to offer a quick overview of the evaluation process and can be easily adapted to different datasets and models.

## Usage

To use the scripts and notebooks, ensure you have the required dependencies installed. These typically include `numpy`, `scipy`, `sklearn`, `umap-learn`, `tqdm`, and `torch` for GPU-accelerated computations. Adjust the paths and parameters in the scripts according to your setup. For your convenience, we uploaded **all inference data (20.9GB) for the evaluation process [here](https://polybox.ethz.ch/index.php/s/xMmdMaccXFt8cqT)!**

For detailed instructions on how to execute each script or notebook, please take a look at the comments within the files. They provide insights into the functionality and expected inputs/outputs.

## Note

The `evaluation.ipynb` notebook is a great starting point for understanding how to apply these tools in practice. It ties together the functions provided in `utilAnalysis.py` and demonstrates how to interpret the results effectively.
