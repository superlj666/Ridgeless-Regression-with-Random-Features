from core_functions.auto_kernel_learning import askl,feature_mapping, initial_weights, linear_regression, Gaussian_kernel
import core_functions.optimal_parameters as optimal_parameters
import os
import logging
import numpy as np
import pickle
import math
from core_functions import utils
import torch

CUDA_LAUNCH_BLOCKING=1
TUNE = False
logger = logging.getLogger('Tune Hyperparamters')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
datasets = ['vowel']
#['dna', 'letter', 'pendigits', 'segment', 'satimage', 'usps', 'shuttle', 'svmguide2', 'vehicle', 'vowel', 'wine','Sensorless', 'mnist']
n_repeats = 10

acc_arr = np.empty((len(datasets), 5, n_repeats))

for (i, dataset) in enumerate(datasets):
    record_ = {
        'KRR' : [],
        'KRidgeless' : [],
        'RF' : [],
        'RF_SGD' : [],
        'RFTK' : []
    }
    
    parameter_dic = optimal_parameters.get_parameter(dataset)
    k_par  = parameter_dic['std1']
    lambda_reg = parameter_dic['lambda_A']
    M = parameter_dic['D']
    parameter_dic['lambda_A'] = 0
    parameter_dic['validate'] = False
    for t in range(n_repeats):
        trainloader, testloader, testloader, d, K  = utils.load_data(dataset, 100000)
        X_train, y_train = iter(trainloader).next()    
        X_test, y_test = iter(testloader).next()
        X_train, y_train, X_test, y_test = X_train.to(device), y_train.to(device), X_test.to(device), y_test.to(device)
        
        K = 1 if utils.is_regression(dataset) else len(y_test.unique())
        n = X_train.shape[0]
        if K == 1:
            loss_type = 'mse'
        elif K == 2:
            loss_type = 'hinge'
            K = 1
        else:
            loss_type = 'cross_entroy'
        if K > 1:
            y_test = y_test.to(torch.long)
            y_train = torch.nn.functional.one_hot(y_train.to(torch.long), K).to(torch.float32)

        if X_train.shape[0] < 2e4:
            K_train = Gaussian_kernel(X_train, X_train, 1/k_par)
            K_test = Gaussian_kernel(X_test, X_train, 1/k_par)

            ### Kernel Ridge
            y_pred = K_test.matmul(torch.linalg.inv(K_train + lambda_reg * n * torch.eye(n, device=device)).matmul(y_train))
            error = utils.test_measure(y_pred, y_test, K)
            print("KRR ---  Error: {:.2f}".format(error))
            acc_arr[i][0][t] = 100 - 100 * error
            record_['KRR'].append(error)

            ### Kernel Ridgeless
            y_pred = K_test.matmul(torch.linalg.inv(K_train + 1e-8 * n * torch.eye(n, device=device)).matmul(y_train))
            error = utils.test_measure(y_pred, y_test, K)
            print("KRR Ridgeless ---  Error: {:.2f}".format(error))
            acc_arr[i][1][t] = 100 - 100 * error
            record_['KRidgeless'].append(error)
        else:
            acc_arr[i][0][t] = 0
            acc_arr[i][1][t] = 0

        ### RF
        d = X_train.shape[1]
        spectral_measure, W = initial_weights(d, M, K, bp = False, std1 = k_par, std2 = 0, stationary=True, device = device, dtype = torch.float)
        Phi_X_train = feature_mapping(X_train, spectral_measure, True)
        Phi_X_test = feature_mapping(X_test, spectral_measure, True)
        RF_W = linear_regression(Phi_X_train, y_train, M, 1e-7)
        y_pred = torch.matmul(Phi_X_test, RF_W)
        error = utils.test_measure(y_pred, y_test, K)
        print("RF Ridgeless ---  Error: {:.2f}".format(error))
        acc_arr[i][2][t] = 100 - 100 * error
        record_['RF'].append(error)
        
        ### RF-SGD
        parameter_dic_SK = parameter_dic.copy()
        parameter_dic_SK['stationary'] = True
        parameter_dic_SK['back_propagation'] = False
        parameter_dic_SK['regular_type'] = 'fro'
        if dataset == 'mnist':
            parameter_dic_SK['learning_rate'] = 1e-2
        RF_SGD_res = askl(parameter_dic_SK)
        acc_arr[i][3][t] = RF_SGD_res['test_accuracy']
        record_['RF_SGD'].append(RF_SGD_res)

        ### RFTK
        parameter_dic_SKL = parameter_dic.copy()
        parameter_dic_SKL['stationary'] = True
        parameter_dic_SKL['back_propagation'] = True
        parameter_dic_SKL['regular_type'] = 'fro'
        RFTK_res = askl(parameter_dic_SKL)
        acc_arr[i][4][t] = RFTK_res['test_accuracy']
        record_['RFTK'].append(RFTK_res)

    with open('{}/exp4_{}.pkl'.format('./results', dataset), 'wb') as handle:
        pickle.dump(record_, handle)
with open('{}/exp4_acc.pkl'.format('./results', dataset), 'wb') as handle:
    pickle.dump(acc_arr, handle)

print(acc_arr)
for (i, dataset) in enumerate(datasets):
    print('{}  &{:.2f}$\pm${:.2f}   &{:.2f}$\pm${:.2f}  &{:.2f}$\pm${:.2f}  &{:.2f}$\pm${:.2f}  &{:.2f}$\pm${:.2f}'.format(dataset, acc_arr[i][0].mean(), acc_arr[i][0].std(), 
    acc_arr[i][1].mean(), acc_arr[i][1].std(),
    acc_arr[i][2].mean(), acc_arr[i][2].std(),
    acc_arr[i][3].mean(), acc_arr[i][3].std(), 
    acc_arr[i][4].mean(), acc_arr[i][4].std()))
    