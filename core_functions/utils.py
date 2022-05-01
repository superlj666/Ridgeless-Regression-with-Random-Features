import torch
import torchvision
import torchvision.transforms as transforms
from torch.distributions.uniform import Uniform
import matplotlib.pyplot as plt
import numpy as np
import os
import math
from sklearn.datasets import load_svmlight_file
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def is_regression(dataest):
    if dataest == 'abalone' or dataest == 'abalone' or dataest == 'cadata' or dataest == 'cpusmall' or dataest == 'space_ga' :
        return True

class svmlight_data(Dataset):
    def __init__(self, data_name, root_dir='./datasets/', transform=None, target_transform=None):
        self.inputs, self.outputs = load_svmlight_file(root_dir + data_name)
        if len(set(self.outputs))  > 2 and self.outputs.min() > 0:
            self.outputs -= self.outputs.min()
        if is_regression(data_name):
            self.outputs = 100*(self.outputs - self.outputs.min()) / (self.outputs.max() - self.outputs.min())
        #self.inputs = self.inputs.toarray()
        #self.outputs = self.outputs.toarray()
        self.transform = transform
        self.target_transform = target_transform
        self.data_name = data_name

    def __len__(self):
        return len(self.outputs)

    def __getitem__(self, idx):
        sample = torch.tensor(self.inputs[idx].A, dtype=torch.float32).view(-1)
        if is_regression(self.data_name):
            label = torch.tensor(self.outputs[idx], dtype=torch.float32)
        else:
            label = torch.tensor(self.outputs[idx], dtype=torch.long)
        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            label = self.target_transform(label)
        return sample, label

def data_tf(x):
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5 # 标准化处理
    x = x.reshape((-1,)) # 拉平
    x = torch.tensor(x)
    return x

def load_data(data_name, batch_size):
    if data_name == 'CIFAR10':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=data_tf)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                              download=True, transform=data_tf)
        trainset, validateset = torch.utils.data.random_split(trainset, [45000, 5000])
        validateloader = torch.utils.data.DataLoader(validateset, batch_size=5000, shuffle=True)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=10000,
                                                 shuffle=False)
        return trainloader, validateloader, testloader, 3072, 10
    elif data_name == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                                download=True, transform=data_tf)
        testset = torchvision.datasets.MNIST(root='./data', train=False,
                                               download=True, transform=data_tf)
        trainset, validateset = torch.utils.data.random_split(trainset, [54000, 6000])
        validateloader = torch.utils.data.DataLoader(validateset, batch_size=6000, shuffle=True)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=10000,
                                                 shuffle=False)
        return trainloader, validateloader, testloader, 784, 10
    elif data_name == "synthetic_nonlinear":
        X_train, y_train, X_test, y_test, _, _ = generate_synthetic(0, 0, 10, 10000, 1)
        train_data = torch.utils.data.TensorDataset(X_train, y_train)
        test_data = torch.utils.data.TensorDataset(X_test, y_test)
        trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
        testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, pin_memory=True)
        return trainloader, testloader, testloader, 10, 1
    else:
        trainset = svmlight_data(data_name)
        trainset, testset = torch.utils.data.random_split(trainset, [int(len(trainset)*0.8), len(trainset) - int(len(trainset)*0.8)])
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=True)
        feature_size = trainset.dataset.inputs.shape[1]
        class_size = len(set(trainset.dataset.outputs))
        return trainloader, testloader, testloader, feature_size, 1 if is_regression(data_name) else class_size

### on all data
def test_measure(outputs, target, K):
    if K > 1:
        _, predicted = torch.max(outputs.data, 1)
        error = (predicted != target).sum().item() / outputs.shape[0]
    else:
        error = empirical_loss(outputs, target, 'mse').item()
    return error

def comp_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res 

def empirical_loss(y_pred, y, loss_type):
    if loss_type == 'mse':
        criterion = torch.nn.MSELoss()
        return criterion(y_pred, y)
    elif loss_type == 'hinge':
        y_pred = torch.tensor([1 if y > 0 else -1 for y in y_pred], dtype=torch.float)
        print(y_pred.shape, y.shape)
        criterion = torch.nn.MultiMarginLoss()
        return criterion(y_pred, y)
    elif loss_type == "cross_entroy":
        criterion = torch.nn.CrossEntropyLoss()
        return criterion(y_pred, y)
    else:
        print('Undefined loss function!\n')
        os._exit()    

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def generate_synthetic(alpha, beta, d, local_size, partitions):
    if local_size == 0:
        samples_per_user = np.random.lognormal(4, 2, partitions).astype(int) + 50
    else:
        samples_per_user = np.zeros(partitions).astype(int) + local_size
    print('>>> Sample per user: {}'.format(samples_per_user.tolist()))

    num_train = sum(samples_per_user)
    num_test = int(num_train/4)
    X_train = np.zeros((partitions, local_size, d))
    y_train = np.zeros((partitions, local_size))

    # prior for parameters
    u = np.random.normal(0, alpha, partitions)
    v = np.random.normal(0, beta, partitions)

    # testing data from the global distribution
    X_test = np.random.multivariate_normal(np.zeros(d), np.eye(d), num_test)
    w_target = np.ones(d)
    y_test = np.min([-np.dot(X_test, w_target), -np.dot(X_test, w_target)], axis=0)

    model_hete = 0
    for i in range(partitions):
        xx = np.random.multivariate_normal(np.ones(d)*u[i], np.eye(d), samples_per_user[i])
        ww = np.random.multivariate_normal(np.ones(d), np.eye(d)*v[i])
        yy = np.min([-np.dot(xx, ww), -np.dot(xx, ww)], axis=0) + np.random.normal(0, 0.2, samples_per_user[i])
        yy_target = np.min([-np.dot(xx, w_target), -np.dot(xx, w_target)], axis=0)
        model_hete += np.linalg.norm(yy - yy_target) / num_train

        X_train[i] = xx
        y_train[i] = yy
        # print("{}-th users has {} exampls".format(i, len(y_split[i])))

    
    data_hete = 0
    X_train_global = X_train.reshape(-1, d)
    C_global = np.matmul(X_train_global.T, X_train_global) / X_train_global.shape[0]
    for i in range(partitions):
        C_local = np.matmul(X_train[i].T, X_train[i]) / X_train[i].shape[0]
        data_hete += np.linalg.norm(C_global - C_local) / partitions

    print("Data heterogeneity: {}, model heterogeneity: {}".format(data_hete, model_hete))

    X_train = torch.tensor(X_train, dtype=torch.float32, device=device).reshape(-1, d)
    X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, dtype=torch.float32, device=device).reshape(-1)
    y_test = torch.tensor(y_test, dtype=torch.float32, device=device)

    return X_train, y_train, X_test, y_test, data_hete, model_hete

def generate_synthetic_binary(alpha, beta, d, local_size, partitions):
    if local_size == 0:
        samples_per_user = np.random.lognormal(4, 2, partitions).astype(int) + 50
    else:
        samples_per_user = np.zeros(partitions).astype(int) + local_size
    print('>>> Sample per user: {}'.format(samples_per_user.tolist()))

    num_train = sum(samples_per_user)
    num_test = int(num_train/4)
    X_train = np.zeros((partitions, local_size, d))
    y_train = np.zeros((partitions, local_size))

    # prior for parameters
    u = np.random.normal(0, alpha, partitions)
    v = np.random.normal(0, beta, partitions)

    # testing data from the global distribution
    X_test = np.random.multivariate_normal(np.zeros(d), np.eye(d), num_test)
    w_target = np.ones(d)
    y_test = np.sign(np.min([-np.dot(X_test, w_target), -np.dot(X_test, w_target)], axis=0))

    model_hete = 0
    for i in range(partitions):
        xx = np.random.multivariate_normal(np.ones(d)*u[i], np.eye(d), samples_per_user[i])
        ww = np.random.multivariate_normal(np.ones(d), np.eye(d)*v[i])
        yy = np.min([-np.dot(xx, ww), -np.dot(xx, ww)], axis=0) + np.random.normal(0, 0.2, samples_per_user[i])
        yy_target = np.min([-np.dot(xx, w_target), -np.dot(xx, w_target)], axis=0)
        model_hete += np.linalg.norm(yy - yy_target) / num_train

        X_train[i] = xx
        y_train[i] = yy
        # print("{}-th users has {} exampls".format(i, len(y_split[i])))

    
    data_hete = 0
    X_train_global = X_train.reshape(-1, d)
    C_global = np.matmul(X_train_global.T, X_train_global) / X_train_global.shape[0]
    for i in range(partitions):
        C_local = np.matmul(X_train[i].T, X_train[i]) / X_train[i].shape[0]
        data_hete += np.linalg.norm(C_global - C_local) / partitions

    print("Data heterogeneity: {}, model heterogeneity: {}".format(data_hete, model_hete))

    X_train = torch.tensor(X_train, dtype=torch.float32, device=device).reshape(-1, d)
    X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, dtype=torch.float32, device=device).reshape(-1)
    y_test = torch.tensor(y_test, dtype=torch.float32, device=device)

    return X_train, y_train, X_test, y_test, data_hete, model_hete