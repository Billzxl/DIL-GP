import argparse

import sklearn
import sklearn.datasets as sk_data
import numpy as np
import random
import torch
import os
from DILGPKernel import DILGPKernel
from kernels import *
from gp import GP
from torch.optim import SGD
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


parser = argparse.ArgumentParser(description='DILGP Tuning')
parser.add_argument('--envlr',
                    help='learning rate for env_w',
                    default=0.001,
                    type=float)
parser.add_argument('--gplr',
                    help='learning rate for gp parameters',
                    default=0.0001,
                    type=float)
parser.add_argument('--epoch',
                    help='learning epoch for gp parameters',
                    default=200,
                    type=int)
parser.add_argument('--eistep',
                    help='number of steps for ei step',
                    default=1,
                    type=int)
parser.add_argument('--lambdae',
                    help='lambda coefficient, loss = marginal_likelihood + npenalty * lambda',
                    default=0,
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
parser.add_argument('--kernel',
                    help='lambda coefficient, loss = marginal_likelihood + npenalty * lambda',
                    default='RationalQuadraticKernel',
                    # default='DotProductKernel',
                    type=str)

# parser.add_argument('--kname',
#                     help='lambda coefficient, loss = marginal_likelihood + npenalty * lambda',
#                     # default='RQ Kernel',
#                     default='DP Kernel',
#                     type=str)
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
model_name = 'dilgp'

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

    return train_data, train_label, valid_data, valid_label


from torch.distributions.normal import Normal
def test(gp, test_data, test_label):
    mu, var = gp.forward(test_data)

    var = torch.clamp(var, min=1e-10, max=1e10)
    std = torch.sqrt(var)
    dis  = Normal(mu.reshape(-1, 1), std.reshape(-1, 1))

    log_ll = dis.log_prob(test_label)

    mu = mu.detach().numpy().flatten()
    std = torch.sqrt(var).detach().numpy().flatten()
    
    # error = np.absolute(mu - test_label.numpy())
    mse_error = np.square(mu - test_label.numpy().flatten())
    error = np.sqrt(mse_error.mean())

    test_label = test_label.numpy().flatten()
    mu_up = mu + std
    mu_down = mu - std
    smaller_than_up = (test_label < mu_up)
    larger_than_down = (test_label > mu_down)

    ratio = ((smaller_than_up & larger_than_down) * 1.0).sum() / len(test_label)

    return error, std.mean(), ratio, log_ll.mean().detach().cpu().numpy()


from matplotlib.transforms import Bbox
def test_plot(gp, train_data, train_label, test_data, test_label, error):
    grid = torch.linspace(-3, 8, 200)[:,None]

    test_data = np.clip(test_data, a_min=-3, a_max=8)

    mu, var = gp.forward(grid)
    mu = mu.detach().numpy().flatten()
    std = torch.sqrt(var).detach().numpy().flatten()

    plt.rcParams.update({'font.size': 20})
    plt.subplots(figsize=(8, 3))
    
    plt.plot(grid.flatten(), mu, color='purple', linewidth=2,alpha=1)

    
    plt.fill_between(grid.flatten(), y1=mu+std, y2=mu-std, alpha=0.5, color='mediumslateblue')
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
    
    # plt.fill_between(grid.flatten(), y1=mu+std, y2=mu-std, alpha=0.3, color='y')

    if model_name == 'dilgp':
        # plt.title(f'GP with {opt.kname}, RMSE={error:.4f}')
        plt.text(4.7,2,f'GP-{opt.kernel} \nRMSE={error:.4f}')
    else:
        plt.title(f'GP, RMSE={error:.4f}')

    plt.savefig(f'result/gp_{opt.kernel}_result.png', bbox_inches=Bbox.from_bounds(0.99, 0.31, 6.23, 2.34))

    plt.cla()




def main():
    global opt
    train_data, train_label, valid_data, valid_label = get_dataset(dataset_name)


    plt.rcParams.update({'font.size': 18})
    setup_seed(opt.seed)

    kernel = eval(opt.kernel)()

    gp = DILGPKernel(kernel, opt.envlr, opt.eistep, opt.lambdae, opt.usekmeans,)#.cuda()

    optimizer = SGD(gp.parameters(), lr=opt.gplr)

    gp.fit(train_data, train_label)

    for i in tqdm(range(opt.epoch)):
        d_train = gp.train_step(train_data, train_label, optimizer)

    with torch.no_grad():
        test_error, test_std, ratio, ll = test(gp, valid_data, valid_label)


    test_plot(gp, train_data, train_label, valid_data, valid_label, test_error)


    print('Model: gp with',opt.kernel)
    print('RMSE: ',test_error)
    print('Coverage Rate: ', ratio)


if __name__ == "__main__":
    main()

