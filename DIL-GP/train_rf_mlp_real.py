import argparse

import sklearn
import sklearn.datasets as sk_data
import numpy as np
import random
import torch
import os
from DILGP import DILGP
from gp import GP
from torch.optim import SGD
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from get_my_data import get_dataset

parser = argparse.ArgumentParser(description='DILGP Tuning')
parser.add_argument('--model_name',
                    help='choose model, rf or mlp',
                    default='rf',
                    choices= ['rf','mlp'],
                    type=str)
parser.add_argument('--envlr',
                    help='learning rate for env_w',
                    default=0.001,
                    type=float)
parser.add_argument('--gplr',
                    help='learning rate for gp parameters',
                    default=0.001,
                    type=float)
parser.add_argument('--epoch',
                    help='learning epoch for gp parameters',
                    default=200,
                    type=int)
parser.add_argument('--eistep',
                    help='number of steps for ei step',
                    default=3,
                    type=int)
parser.add_argument('--lambdae',
                    help='lambda coefficient, loss = marginal_likelihood + npenalty * lambda',
                    default=0.5,
                    type=float)
parser.add_argument('--seed',
                    help='random seed',
                    default=0,
                    type=int)
parser.add_argument('--usekmeans',
                    help='whether use kmeans to initialize env_w',
                    default=True,
                    type=bool)
parser.add_argument('--length_scale',
                    help='initialization of length_scale',
                    default=1.0,
                    type=float)
parser.add_argument('--noise_scale',
                    help='initialization of noise_scale',
                    default=1.0,
                    type=float)
parser.add_argument('--amplitude_scale',
                    help='initialization of amplitude_scale',
                    default=1.0,
                    type=float)
opt = parser.parse_args()


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


dataset_name = 'auto_mobile'

model_name = opt.model_name



def test(gp, test_data, test_label):
    mu = gp.predict(test_data).flatten()

    # error = np.absolute(mu - test_label.numpy())
    mse_error = np.square(mu - test_label.flatten())
    error = np.sqrt(mse_error.mean())

    return error


def test_plot(gp, train_data, train_label, test_data, test_label, error):
    grid = torch.linspace(-3, 8, 200)[:,None]

    test_data = np.clip(test_data, a_min=-3, a_max=8)

    mu = gp.predict(grid)
    mu = mu.flatten()
    # std = torch.sqrt(var).detach().numpy().flatten()

    plt.rcParams.update({'font.size': 18})
    plt.subplots(figsize=(8, 6))
    plt.tight_layout()
    plt.plot(grid.flatten(), mu, color='blue')
    plt.plot(train_data.flatten(), train_label, '.', color='green')
    plt.plot(test_data.flatten(), test_label, '.', color='yellow')
    plt.plot(grid.flatten(), mu)


    # plt.title(f'After hyperparameter optimization, error={error:.4f}')
    # if model_name == 'dilgp':
    #     plt.savefig(f'debug/train_dilgp_result.png')
    # else:
    #     plt.savefig(f'debug/train_gp_result.png')
    if model_name == "mlp":
        plt.title(f'MLP, error={error:.4f}')
        plt.savefig(f'debug/train_mlp_result.png')
    else:
        plt.title(f'RandomForest, error={error:.4f}')
        plt.savefig(f'debug/train_regr_result.png')

    plt.cla()


def main():
    global opt
    train_data, train_label, valid_data, valid_label,_ = get_dataset(dataset_name)
    train_data, train_label, valid_data, valid_label = \
        train_data.numpy(), train_label.numpy(), valid_data.numpy(), valid_label.numpy()

    setup_seed(opt.seed)

    random_state = 3
        
    if model_name == "mlp":
        regr = MLPRegressor(hidden_layer_sizes=(64, 64, 64), random_state=random_state, max_iter=5000).fit(train_data, train_label)
    else:
        regr = RandomForestRegressor(n_estimators=50, random_state=random_state)

    regr.fit(train_data, train_label)

    test_error = test(regr, valid_data, valid_label)

    print('Model: ',opt.model_name)
    print('RMSE: ',test_error)   



if __name__ == "__main__":
    main()

