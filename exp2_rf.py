import enum
from numpy.core.function_base import linspace
import core_functions.utils as utils
from core_functions.auto_kernel_learning import feature_mapping, initial_weights, linear_regression, Gaussian_kernel
import os
import pickle
import argparse
import logging
import nni
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from nni.utils import merge_parameter
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
    stationary = True
    back_propagation =False
    k_par = 0.1 #param['kernel_par'] # 0.1 for kernel
    lambda_reg = 0.000001 #param['lambda_reg'] # 0.000001 for kernel
    M = 1000 #param['hidden_size']
    d = 10
    K = 1
    loss_type = 'mse'
    T = 20
    threshold = 0.006    
    data_dir = './data'
    result_dir = './results'
    repeat = 100
    n = 100

    lambda_arr = [1e-7, 1e-4, 1e-2, 1]
    M_arr = (np.logspace(-6.2, 0, base=4, num=50) * 10000).astype(int)

    kernel_error = np.ones((len(lambda_arr), len(M_arr), repeat))
    rf_error = np.empty((len(lambda_arr), len(M_arr), repeat))
    approx_arr = np.empty((len(lambda_arr), len(M_arr), repeat))
    M_n_arr = M_arr / n
    for r in range(repeat):
        trainloader, validateloader, testloader, d, K = utils.load_data('mnist', 100)

        X_train, y_train = iter(trainloader).next()    
        X_test, y_test = iter(testloader).next()
        X_train, y_train, X_test, y_test = X_train.to(device), torch.nn.functional.one_hot(y_train, K).to(device), X_test.to(device), y_test.to(device)

        #### KRR
        for (i, lambda_reg) in enumerate(lambda_arr):
            K_train = Gaussian_kernel(X_train, X_train, 1/k_par)
            K_test = Gaussian_kernel(X_test, X_train, 1/k_par)
            y_pred = K_test.matmul(torch.linalg.inv(K_train + lambda_reg * n * torch.eye(n, device=device)).matmul(y_train.float()))
            kernel_loss = utils.empirical_loss(y_pred, torch.nn.functional.one_hot(y_test, K), loss_type).item()
            error = 1 - utils.comp_accuracy(y_pred, y_test)[0].item()/100
            kernel_error[i, :, r] *= error
            logger.info("KRR --- MSE: {} Error: {:.2f}".format(kernel_loss, error))
            if TUNE:
                nni.report_final_result(error)

        ## RF
        for (i, lambda_reg) in enumerate(lambda_arr):
            for (j, M) in enumerate(M_arr):
                spectral_measure, W = initial_weights(d, M, K, bp = False, std1 = k_par, std2 = 0, stationary=stationary, device = device, dtype = torch.float)
                Phi_X_train = feature_mapping(X_train, spectral_measure, stationary)
                Phi_X_test = feature_mapping(X_test, spectral_measure, stationary)
                RF_W = linear_regression(Phi_X_train, y_train, M, lambda_reg)
                y_pred = torch.matmul(Phi_X_test, RF_W)
                kernel_loss = utils.empirical_loss(y_pred, torch.nn.functional.one_hot(y_test, K), loss_type).item()
                error = 1 - utils.comp_accuracy(y_pred, y_test)[0].item()/100
                rf_error[i, j, r] = error
                logger.info("RFF --- MSE: {} Error: {:.2f}".format(kernel_loss, error))

                K_phi = Phi_X_train.mm(Phi_X_train.T)
                approx_arr[i, j, r] = torch.norm(K_train - K_phi).item() / (X_train.shape[0] * X_train.shape[0])

    record_ = {
                'lambda_arr': M_arr,
                'M_n_arr': M_n_arr,
                'kernel_error': kernel_error,
                'rf_error': rf_error,
                'approx_arr': approx_arr,
                'name': ['$\widehat f$', '$\widehat f_\lambda, \lambda = 10^{-2}$', '$\widehat{f}_\lambda, \lambda = 10^{-4}$', '$\widehat f_\lambda, \lambda = 10^{-6}$', '\widehat f_M']
        }

    result_path = '{}/exp2_{}.pkl'.format(result_dir, 'mnist')
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
    
