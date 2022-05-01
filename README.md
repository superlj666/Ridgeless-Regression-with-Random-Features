# Ridgeless Regression with Random Features
## Intro
This repository provides the code used to run the experiments of the paper "Ridgeless Regression with Random Features".
## Environments
- Python 3.7.4
- Pytorch 1.10.0
- NNI 2.5
- CUDA 10.1.168
- cuDnn 7.6.0
- GPU: Nvidia RTX 2080Ti 11G
## Core functions
- auto_kernel_learning.py implements the algorithm to construct an one-layer neural network, including initialization of trainable weights and untrainable biases as well as feature mapping (cosine as activation).
- utils.py implements useful tools including load svmlight style dataset and classic datasets used in Pytorch but also various loss functions are introduced.
- parameter_tune.py is used to tune hyperparameters via [NNI](https://nni.readthedocs.io/).
- optimal_parameters.py records optimal parameters for the proposed algorithm.
- exp1_rf_sgd.py, exp2_rf.py, exp3_trace.py and exp4_real.py are scripts in experiments.
- plot.ipynb reads experiment results and draw images.
## Experiments
1. Download datasets for multi-class classification (https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/).
2. Run the script to tune parameters and record them in optimal_parameters.py.
```
python parameter_tune.py
```
3. Run the scripts to obtain results in Experiment section
```
python exp1_rf_sgd.py
python exp2_rf.py
python exp3_trace.py
python exp4_real.py
```
4. Run plot.ipynb to draw figures
