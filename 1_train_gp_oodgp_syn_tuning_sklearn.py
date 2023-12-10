import argparse

import sklearn
import sklearn.datasets as sk_data
import numpy as np
import random
import torch
import os
from OODGP import OODGP
from gp import GP
from torch.optim import SGD
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor


parser = argparse.ArgumentParser(description='OODGP Tuning')
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

# setup_seed(0)
dataset_name = 'synthetic'
model_name = 'mlp'
# model_name = 'rf'


def get_dataset(dataset_name):
    setup_seed(0)
    X = torch.randn(100,1)
    f = torch.sin(X * 2 * np.pi /4).flatten()
    y = f + torch.randn_like(f) * 0.1
    y = y[:,None] * 3

    X2 = torch.randn(100,1) + 6.5
    f = -torch.sin((X2-6.5) * 2 * np.pi / 64).flatten()
    y2 = f + torch.randn_like(f) * 0.1 + 0.5
    y2 = y2[:,None]

    train_data = torch.cat([X, X2[:15]], dim=0)
    train_label = torch.cat([y, y2[:15]], dim=0)
    valid_data = X2[20:]
    valid_label = y2[20:]

    return train_data.numpy(), train_label.numpy(), valid_data.numpy(), valid_label.numpy()


def test(gp, test_data, test_label):
    mu = gp.predict(test_data).flatten()

    # error = np.absolute(mu - test_label.numpy())
    mse_error = np.square(mu - test_label.flatten())
    error = np.sqrt(mse_error.mean())

    return error

from matplotlib.transforms import Bbox
def test_plot(gp, train_data, train_label, test_data, test_label, error):
    grid = torch.linspace(-3, 8.5, 200)[:,None]

    test_data = np.clip(test_data, a_min=-3, a_max=8)

    mu = gp.predict(grid)
    mu = mu.flatten()
    # std = torch.sqrt(var).detach().numpy().flatten()

    plt.rcParams.update({'font.size': 20})
    plt.subplots(figsize=(8, 3))
    
    plt.plot(grid.flatten(), mu, color='purple', linewidth=2,alpha=1)

    
    # plt.fill_between(grid.flatten(), y1=mu+std, y2=mu-std, alpha=0.5, color='mediumslateblue')
    # plt.scatter(train_data.flatten(), train_label, 
    #             c=np.array((64, 14, 50)).reshape(1, 3)/255, s=25)
    plt.scatter(test_data.flatten(), test_label, 
             c='orange', s=15)
    plt.scatter(train_data.flatten(), train_label, 
                c='blue', s=15)

    # plt.scatter(test_data.flatten(), test_label, 
    #          c=np.array((242, 102, 171)).reshape(1, 3)/255, s=25)
    # plt.title(f'After hyperparameter optimization, error={error:.4f}')
    plt.xticks([])
    plt.yticks([])
    # plt.subplots(figsize=(8, 5))
    # plt.tight_layout()
    # plt.plot(grid.flatten(), mu, color='orange', linewidth=2)
    # plt.scatter(train_data.flatten(), train_label, 
    #             c=np.array((64, 14, 50)).reshape(1, 3)/255, s=25)
    # plt.scatter(test_data.flatten(), test_label, 
    #          c=np.array((242, 102, 171)).reshape(1, 3)/255, s=25)

    # plt.title(f'After hyperparameter optimization, error={error:.4f}')
    # if model_name == 'oodgp':
    #     plt.savefig(f'debug/train_oodgp_result.png')
    # else:
    #     plt.savefig(f'debug/train_gp_result.png')
    if model_name == "mlp":
        # plt.title(f'MLP, RMSE={error:.4f}')
        plt.text(3.3,3.4,f'MLP, RMSE={error:.4f}')
        plt.savefig(f'debug/6_train_mlp_result.png', bbox_inches=Bbox.from_bounds(0.99, 0.31, 6.23, 2.34))
    else:
        # plt.title(f'RandomForest, RMSE={error:.4f}')
        plt.text(3.7,3,f'RF, RMSE={error:.4f}')
        plt.savefig(f'debug/6_train_regr_result.png',  bbox_inches=Bbox.from_bounds(0.99, 0.31, 6.23, 2.34))

    plt.cla()


def main():
    global opt
    train_data, train_label, valid_data, valid_label = get_dataset(dataset_name)

    setup_seed(opt.seed)

    error_diff_seed = []
    error0 = None
    for i in range(10):
        if model_name == "mlp":
            regr = MLPRegressor(hidden_layer_sizes=(64, 64, 64), random_state=i, max_iter=5000).fit(train_data, train_label)
            # regr = MLPRegressor(random_state=1, max_iter=500,hidden_layer_sizes=).fit(train_data, train_label)
        else:
            regr = RandomForestRegressor(n_estimators=50, random_state=i)

        regr.fit(train_data, train_label)

        l_error = [];
        test_error = test(regr, valid_data, valid_label)
        l_error.append(test_error.mean())

        error = l_error[-1]

        fig, axs = plt.subplots(ncols=5, figsize=(20,4))
        axs[4].plot(np.stack(l_error)); axs[4].set_title('Valid Error'); axs[4].set_xlabel('iteration');


        if model_name == "mlp":
            plt.savefig('debug/train_mlp.png')
        else:
            plt.savefig('debug/train_gp_regr.png')
        
        plt.cla()

        error_diff_seed.append(error)
        print(error)

        if (model_name == "mlp" and i == 1) or (model_name != "mlp" and i == 0):
            error0  = error
            test_plot(regr, train_data, train_label, valid_data, valid_label, error)

    print(error_diff_seed)
    diff = np.abs(np.array(error_diff_seed) - error0)
    print(error0, diff.max())

    




if __name__ == "__main__":
    main()

