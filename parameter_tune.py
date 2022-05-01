"""
Tune Hyperparamters for KRR via NNI.
"""

from core_functions.auto_kernel_learning import askl,feature_mapping, initial_weights, linear_regression, Gaussian_kernel
import core_functions.optimal_parameters as optimal_parameters
import os
import logging
import numpy as np
from nni.utils import merge_parameter
import pickle
import argparse
import math
from core_functions import utils
import torch
import nni

CUDA_LAUNCH_BLOCKING=1
SAVE = False
logger = logging.getLogger('Tune Hyperparamters')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    torch.manual_seed(args['seed'])
    np.random.seed(args['seed'])
    dataset = args['dataset']
    num_classes = args['num_classes']
    kernel_type = args['kernel_type']
    epochs = args['epochs']
    k_par = args['kernel_par']
    task_type = args['task_type']
    lambda_reg = args['lambda_reg']
    data_dir = args['data_dir']
    D = args['hidden_size']
    local_step = int(args['local_step'])
    lr = args['lr']
    alpha = args['alpha']
    beta = args['beta']
    d = args['d']

    # trainset = utils.svmlight_data(dataset)
    # data_size = trainset.inputs.shape[0]
    # inputs = trainset.inputs.toarray()
    # X_train = torch.tensor(inputs[:int(data_size*0.8),:], dtype=torch.float32, device=device)
    # X_test = torch.tensor(inputs[int(data_size*0.8):,:], dtype=torch.float32, device=device)
    # y_train = torch.tensor(trainset.outputs[:int(data_size*0.8)], dtype=torch.float32, device=device)
    # y_test = torch.tensor(trainset.outputs[int(data_size*0.8):], dtype=torch.float32, device=device)
    # K_train = Gaussian_kernel(X_train, X_train, 1/k_par)
    # K_test = Gaussian_kernel(X_test, X_train, 1/k_par)
    # n = X_train.shape[0]
    # K = 1 if utils.is_regression(dataset) else len(y_test.unique())
    # if K == 1:
    #     loss_type = 'mse'
    # elif K == 2:
    #     loss_type = 'hinge'
    #     K = 1
    # else:
    #     loss_type = 'cross_entroy'
    # if K > 1:
    #     y_test = y_test.to(torch.long)
    #     y_train = torch.nn.functional.one_hot(y_train.to(torch.long), K).to(torch.float32)

    # y_pred = K_test.matmul(torch.linalg.inv(K_train + lambda_reg * n * torch.eye(n, device=device)).matmul(y_train))
    # error = utils.test_measure(y_pred, y_test, K)
    # logger.info("KRR --- Error: {:.2f}".format(error))

    parameter_dic_SKL = optimal_parameters.get_parameter(dataset)
    parameter_dic_SKL['stationary'] = True
    parameter_dic_SKL['back_propagation'] = True
    parameter_dic_SKL['regular_type'] = 'fro'
    parameter_dic_SKL['std1'] = params['kernel_par']
    parameter_dic_SKL['lambda_A'] = 0
    parameter_dic_SKL['lambda_B'] = params['lambda_reg']
    parameter_dic_SKL['learning_rate'] = params['lr']
    parameter_dic_SKL['T'] = 1
    res = askl(parameter_dic_SKL)
    error = 1 - res['test_accuracy'] / 100.

    nni.report_final_result(error)
    logger.debug('Final result is %.4f', 
    error)
    logger.debug('Send final result done.')

def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description='KRR Tuner')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument("--dataset", type=str,
                        default='mnist', help="dataset name")
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

## set parameters in config_gpu.yml 
## and run nnictl create --config /home/superlj666/Experiment/FedNewton/config_gpu.yml
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