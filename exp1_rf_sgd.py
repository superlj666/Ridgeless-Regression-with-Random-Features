"""
Tune Hyperparamters for KRR via NNI.
"""

import core_functions.utils as utils
from core_functions.auto_kernel_learning import feature_mapping, initial_weights, linear_regression
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

def RF_SGD(learner, train_data, X_test, y_test, spectral_measure, W, RF_W, threshold):
    batch_size = learner['batch_size']
    learning_rate = learner['learning_rate']
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=False)
    optimizer = optim.Adam((spectral_measure[0], W), learning_rate)

    start = time.time()

    # Records variables
    training_loss_records = []
    validate_loss_records = []
    difference_records = []

    for epoch in range(learner['T']):
        for i_batch, train_batch in enumerate(trainloader, 0):
            optimizer.zero_grad() 
            X_train, y_train = train_batch
            
            # Forward : predict 
            y_pred = X_train.mm(W)
            
            # Forward : calculate objective
            loss = utils.empirical_loss(y_pred, y_train, 'mse')

            # Backward
            loss.backward()

            # Update
            optimizer.step()
        
        # Record for each epoch
        difference_records.append(torch.norm(W - RF_W).item()/X_test.shape[1])
        training_loss_records.append(loss.item())
        y_pred_val = X_test.mm(W)            
        validate_loss_records.append(utils.empirical_loss(y_pred_val, y_test,'mse').item())
        if TUNE:
            nni.report_intermediate_result(validate_loss_records[-1])
        if validate_loss_records[-1] < threshold:
            break

    end = time.time()
    print('Elapsed time: {} seconds'.format(end - start))
    return training_loss_records, validate_loss_records, difference_records, end - start

def main(param):
    torch.manual_seed(1)
    np.random.seed(1)
    stationary = True
    back_propagation =False
    k_par = 0.1 #param['kernel_par']
    d = 10
    K = 1
    M = 200
    n = 10000
    loss_type = 'mse'
    lambda_reg = 0
    T = 20
    threshold = 0.006
    
    data_dir = './data'
    result_dir = './results'
    repeat = 1

    ### RF
    X_train, y_train, X_test, y_test, _, _ = utils.generate_synthetic(0, 0, d, n, 1)
    spectral_measure, W = initial_weights(d, M, K, bp = False, std1 = k_par, std2 = 0, stationary=stationary, device = device, dtype = torch.float)
    Phi_X_train = feature_mapping(X_train, spectral_measure, stationary)
    Phi_X_test = feature_mapping(X_test, spectral_measure, stationary)
    RF_W = linear_regression(Phi_X_train, y_train, M, lambda_reg).reshape(-1, 1)
    y_pred = torch.matmul(Phi_X_test, RF_W)
    rf_loss = utils.empirical_loss(y_pred, y_test, loss_type).item()
    logger.info("RFF --- MSE: {}".format(rf_loss))

    # ### RF-SGD
    learners = [
        {'batch_size': 1, 'learning_rate': 0.001, 'T' : 100},
        {'batch_size': 10, 'learning_rate': 0.001, 'T' : 100},
        {'batch_size': 100, 'learning_rate': 0.01, 'T' : 100},
        {'batch_size': 1000, 'learning_rate': 0.1, 'T' : 100}, # 1000-0.1
        {'batch_size': 10000, 'learning_rate': 0.1, 'T' : 100},
        ]

    data_ = []
    train_data = torch.utils.data.TensorDataset(Phi_X_train, y_train)
    for learner in learners:
        W_tmp = W.clone().detach().requires_grad_(True)
        training_loss_records, validate_loss_records, difference_records, training_time = RF_SGD(learner, train_data, Phi_X_test, y_test, spectral_measure, W_tmp, RF_W, threshold)
        record_ = {
            'training_loss_records': training_loss_records,
            'validate_loss_records': validate_loss_records,
            'difference_records': difference_records,
            'training_time': training_time
        }
        data_.append(record_)

    result_path = '{}/exp1_{}.pkl'.format(result_dir, 'synthetic_nonlinear')
    with open(result_path, "wb") as f:
        pickle.dump(data_, f)
    
    if TUNE:
        nni.report_final_result(validate_loss_records[-1])
        logger.debug('Final result is %.4f', validate_loss_records[-1])
        logger.debug('Send final result done.')


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
    
