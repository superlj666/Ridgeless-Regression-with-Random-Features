def get_parameter(dataset):
    if dataset == 'mnist':
        # Parameters for mnist
        parameter_dic = {
            'dataset' : 'mnist', #'CIFAR10',
            'D' : 2000,
            'stationary' : False,
            'back_propagation' : True,
            'std1' : 0.1, #0.0001, 
            'std2' : 0.03125,
            'batch_size' : 32, 
            'T' : 10,
            'record_batch' : 200,
            'loss_type' : 'cross_entroy', # 'hinge' 'mse'
            'lambda_A' : 0.0001, 
            'lambda_B' : 100.001, 
            'regular_type' : 'nuc', # 'fro'
            'learning_rate' : 1e-4,#0.001,
            'validate' : True
        }
    elif dataset == 'usps':
        # Parameters for usps
        parameter_dic = {
            'dataset' : 'usps', 
            'D' : 2000,
            'stationary' : False,
            'back_propagation' : True,
            'std1' : 0.01,#0.1, 
            'std2' : 0.125,
            'batch_size' : 32, 
            'T' : 10,
            'record_batch' : 100,
            'loss_type' : 'cross_entroy', # 'hinge' 'mse'
            'lambda_A' : 0.000001, 
            'lambda_B' : 0.01, 
            'regular_type' : 'nuc', # 'fro'
            'learning_rate' : 1e-3,
            'validate' : True
        }
    elif dataset == 'pendigits':
        parameter_dic = {
            'dataset' : 'pendigits', 
            'D' : 2000,
            'stationary' : False,
            'back_propagation' : True,
            'std1' : 0.001,#0.01, 
            'std2' : 0.05,
            'batch_size' : 32, 
            'T' : 10,
            'record_batch' : 100,
            'loss_type' : 'cross_entroy', # 'hinge' 'mse'
            'lambda_A' : 1e-7, 
            'lambda_B' : 1, 
            'regular_type' : 'nuc', # 'fro'
            'learning_rate' : 0.0001,
            'validate' : True
        }
    elif dataset == 'poker':
        parameter_dic = {
            'dataset' : dataset, 
            'D' : 2000,
            'stationary' : False,
            'back_propagation' : True,
            'std1' : 0.1, 
            'std2' : 0.125,
            'batch_size' : 32, 
            'T' : 10,
            'record_batch' : 100,
            'loss_type' : 'cross_entroy', # 'hinge' 'mse'
            'lambda_A' : 0.01, 
            'lambda_B' : 1e-3, 
            'regular_type' : 'nuc', # 'fro'
            'learning_rate' : 0.001,
            'validate' : True
        }
    elif dataset == 'abalone':
        parameter_dic = {
            'dataset' : dataset, 
            'D' : 2000,
            'stationary' : False,
            'back_propagation' : True,
            'std1' : 1, 
            'std2' : 0.125, 
            'batch_size' : 32, 
            'T' : 10,
            'record_batch' : 100,
            'loss_type' : 'cross_entroy', # 'hinge' 'mse'
            'lambda_A' : 0.0001, 
            'lambda_B' : 1e-3, 
            'regular_type' : 'nuc', # 'fro'
            'learning_rate' : 1e-2,
            'validate' : True
        }
    elif dataset == 'space_ga':
        parameter_dic = {
            'dataset' : dataset, 
            'D' : 2000,
            'stationary' : False,
            'back_propagation' : True,
            'std1' : 0.001, 
            'std2' : 0.03125, 
            'batch_size' : 32, 
            'T' : 10,
            'record_batch' : 100,
            'loss_type' : 'cross_entroy', # 'hinge' 'mse'
            'lambda_A' : 0.0001, 
            'lambda_B' : 1e-3, 
            'regular_type' : 'nuc', # 'fro'
            'learning_rate' : 0.001,
            'validate' : True
        }
    elif dataset == 'cpusmall':
        parameter_dic = {
            'dataset' : dataset, 
            'D' : 2000,
            'stationary' : False,
            'back_propagation' : True,
            'std1' : 0.015625, 
            'std2' : 0.015625, 
            'batch_size' : 32, 
            'T' : 10,
            'record_batch' : 100,
            'loss_type' : 'cross_entroy', # 'hinge' 'mse'
            'lambda_A' : 1e-3, 
            'lambda_B' : 1e-3, 
            'regular_type' : 'nuc', # 'fro'
            'learning_rate' : 1e-2,
            'validate' : True
        }
    elif dataset == 'cadata':
        parameter_dic = {
            'dataset' : dataset, 
            'D' : 2000,
            'stationary' : False,
            'back_propagation' : True,
            'std1' : 0.0078125,
            'std2' : 0.0078125, 
            'batch_size' : 32, 
            'T' : 10,
            'record_batch' : 100,
            'loss_type' : 'cross_entroy', # 'hinge' 'mse'
            'lambda_A' : 1e-5, 
            'lambda_B' : 1e-4, 
            'regular_type' : 'nuc', # 'fro'
            'learning_rate' : 1e-2,
            'validate' : True
        }
    elif dataset == 'wine':
        parameter_dic = {
            'dataset' : dataset, 
            'D' : 2000,
            'stationary' : False,
            'back_propagation' : True,
            'std1' : 0.001,#0.01,
            'std2' : 0.0078125, 
            'batch_size' : 32, 
            'T' : 10,
            'record_batch' : 100,
            'loss_type' : 'cross_entroy', # 'hinge' 'mse'
            'lambda_A' : 1, 
            'lambda_B' : 0.001, 
            'regular_type' : 'nuc', # 'fro'
            'learning_rate' : 0.01,
            'validate' : True
        }
    elif dataset == 'Sensorless':
                parameter_dic = {
            'dataset' : dataset, 
            'D' : 2000,
            'stationary' : False,
            'back_propagation' : True,
            'std1' : 0.01, 
            'std2' : 0.05,
            'batch_size' : 32, 
            'T' : 10,
            'record_batch' : 100,
            'loss_type' : 'cross_entroy', # 'hinge' 'mse'
            'lambda_A' : 0.000001, 
            'lambda_B' : 1e-3, 
            'regular_type' : 'nuc', # 'fro'
            'learning_rate' : 0.001,
            'validate' : True
        }
    elif dataset == 'dna':
            parameter_dic = {
            'dataset' : dataset, 
            'D' : 2000,
            'stationary' : False,
            'back_propagation' : True,
            'std1' : 0.0001, 
            'std2' : 0.05,
            'batch_size' : 32, 
            'T' : 10,
            'record_batch' : 100,
            'loss_type' : 'cross_entroy', # 'hinge' 'mse'
            'lambda_A' : 0.00001, 
            'lambda_B' : 100.001, 
            'regular_type' : 'nuc', # 'fro'
            'learning_rate' : 0.001,
            'validate' : True
        }
    elif dataset == 'segment':
        parameter_dic = {
            'dataset' : dataset, 
            'D' : 2000,
            'stationary' : False,
            'back_propagation' : True,
            'std1' : 0.01,#1.01, 
            'std2' : 0.05,
            'batch_size' : 32, 
            'T' : 10,
            'record_batch' : 100,
            'loss_type' : 'cross_entroy', # 'hinge' 'mse'
            'lambda_A' : 1e-6, 
            'lambda_B' : 100.001, 
            'regular_type' : 'nuc', # 'fro'
            'learning_rate' : 0.01,
            'validate' : True
        }
    elif dataset == 'svmguide2':
            parameter_dic = {
            'dataset' : dataset, 
            'D' : 2000,
            'stationary' : False,
            'back_propagation' : True,
            'std1' : 10, 
            'std2' : 0.05,
            'batch_size' : 32, 
            'T' : 10,
            'record_batch' : 100,
            'loss_type' : 'cross_entroy', # 'hinge' 'mse'
            'lambda_A' : 0.0001, 
            'lambda_B' : 0.001, 
            'regular_type' : 'nuc', # 'fro'
            'learning_rate' : 0.01,
            'validate' : True
        }
    elif dataset == 'vehicle':
            parameter_dic = {
            'dataset' : dataset, 
            'D' : 2000,
            'stationary' : False,
            'back_propagation' : True,
            'std1' : 1, 
            'std2' : 0.05,
            'batch_size' : 32, 
            'T' : 10,
            'record_batch' : 100,
            'loss_type' : 'cross_entroy', # 'hinge' 'mse'
            'lambda_A' : 0.1, 
            'lambda_B' : 1e-3, 
            'regular_type' : 'nuc', # 'fro'
            'learning_rate' : 0.01,
            'validate' : True
        }
    elif dataset == 'vowel':
            parameter_dic = {
            'dataset' : dataset, 
            'D' : 2000,
            'stationary' : False,
            'back_propagation' : True,
            'std1' : 0.1,#1.01, 
            'std2' : 0.05,
            'batch_size' : 32, 
            'T' : 10,
            'record_batch' : 100,
            'loss_type' : 'cross_entroy', # 'hinge' 'mse'
            'lambda_A' : 0.00001, 
            'lambda_B' : 0.001,#0.001, 
            'regular_type' : 'nuc', # 'fro'
            'learning_rate' : 0.01,
            'validate' : True
        }
    else:
        parameter_dic = {
            'dataset' : dataset, 
            'D' : 2000,
            'stationary' : False,
            'back_propagation' : True,
            'std1' : 1, 
            'std2' : 0.05,
            'batch_size' : 32, 
            'T' : 10,
            'record_batch' : 100,
            'loss_type' : 'cross_entroy', # 'hinge' 'mse'
            'lambda_A' : 1e-6, 
            'lambda_B' : 1e-3, 
            'regular_type' : 'nuc', # 'fro'
            'learning_rate' : 1e-2,
            'validate' : True
        }
    parameter_dic['local_update'] = 100
    return parameter_dic
