import logging
import nni
import json
import argparse
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from nni.utils import merge_parameter
from core_functions.auto_kernel_learning import askl
from core_functions import optimal_parameters
from torchvision import datasets, transforms


CUDA_LAUNCH_BLOCKING=1
TUNE = False
logger = logging.getLogger('Tune Hyperparamters')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description='KRR Tuner')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument("--dataset", type=str,
                        default='synthetic_nonlinear', help="dataset name")
    parser.add_argument("--alpha", type=float, 
                        default=0.0, help="data hetegeneity parameter")
    parser.add_argument("--beta", type=float, 
                        default=0.0, help="model hetegeneity")
    parser.add_argument("--d", type=int, 
                        default=10, help="data dimensionality")
    parser.add_argument("--task_type", type=str,
                        default='regression', help="task type, including (regression, binary, multiclass")
    parser.add_argument("--num_classes", type=int,
                        default=10, help="classes of multiclass task")
    parser.add_argument("--kernel_type", type=str,
                        default='gaussian', help="kernel type (default: gaussian), or linear")
    parser.add_argument("--kernel_par", type=float, 
                        default=100.01, help="kernel hyperparameter")
    parser.add_argument("--hidden_size", type=int, default=200, metavar='N',
                        help='hidden layer size (default: 200)')
    parser.add_argument('--lambda_reg', type=float, default=0.000001, help='regularizer parameter (default: 0.01)')
    parser.add_argument("--data_dir", type=str,
                        default='./data', help="data directory")
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001)')
    parser.add_argument('--local_step', type=int, default=8, help='local update (default: 1)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    args, _ = parser.parse_known_args()
    return args

def main(param):
    torch.manual_seed(1)
    np.random.seed(1)
    dataset = 'mnist'
    n_repeats = 1
    result_dir = './results'
    results_rf_sgd = []
    results_rf_tk = []
    results_rf_tk_trace = []
    parameter_dic = optimal_parameters.get_parameter(dataset)

    parameter_dic['stationary'] = True
    parameter_dic['std1'] = 0.1
    parameter_dic['lambda_A'] = 0

    parameter_dic['local_update'] = 10
    parameter_dic['lambda_B'] = 100
    parameter_dic['T'] = 30

    for t in range(n_repeats):
        # RF-SGD Ridgeless
        parameter_dic_SK = parameter_dic.copy()
        # parameter_dic_SK['learning_rate'] = 1e-2
        parameter_dic_SK['back_propagation'] = False
        results_rf_sgd.append(askl(parameter_dic_SK))

        # # RFTK Ridgeless
        # parameter_dic_SKL = parameter_dic.copy()
        # parameter_dic_SKL['back_propagation'] = True
        # parameter_dic_SK['lambda_B'] = 0
        # results_rf_tk.append(askl(parameter_dic_SKL))

        # RFTK Ridgeless + trace
        parameter_dic_SKL = parameter_dic.copy()
        parameter_dic_SKL['back_propagation'] = True
        results_rf_tk_trace.append(askl(parameter_dic_SKL))

    record_ = {'results_rf_sgd': results_rf_sgd, 
            'results_rf_tk': results_rf_tk, 
            'results_rf_tk_trace': results_rf_tk_trace}
    result_path = '{}/exp3_{}.pkl'.format(result_dir, 'mnist')
    with open(result_path, "wb") as f:
        pickle.dump(record_, f)

if __name__ == '__main__':
    try:
        # get parameters form tuner
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(merge_parameter(get_params(), tuner_params))
        print(params)
        main(params)

    except Exception as exception:
        logger.exception(exception)
        raise