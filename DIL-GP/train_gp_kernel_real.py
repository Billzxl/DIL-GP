import argparse

import sklearn
import sklearn.datasets as sk_data
import numpy as np
import random
import torch
import os
from DILGP import DILGP
from gp import GP
from DILGPKernel import DILGPKernel
from torch.optim import SGD
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kernels import *
from get_my_data import get_dataset
# from get_housing import get_dataset

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

parser = argparse.ArgumentParser(description='DILGP Tuning')
parser.add_argument('--envlr',
                    help='learning rate for env_w',
                    default=0.01,
                    type=float)
parser.add_argument('--gplr',
                    help='learning rate for gp parameters',
                    default=0.01,
                    type=float)
parser.add_argument('--epoch',
                    help='learning epoch for gp parameters',
                    default=100,
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


dataset_name = "auto_mobile"


model_name = 'gp_kernel'
device = torch.device('cuda:2') # or torch.device('cpu')


from torch.distributions.normal import Normal
def test(gp, test_data, test_label):
    mu, var = gp.forward(test_data)

    var = torch.clamp(var, min=1e-10, max=1e10)
    std = torch.sqrt(var)
    # print(std)
    dis  = Normal(mu.reshape(-1, 1), std.reshape(-1, 1))

    log_ll = dis.log_prob(test_label)

    mu = mu.detach().cpu().numpy().flatten()
    std = torch.sqrt(var).detach().cpu().numpy().flatten()
    
    # error = np.absolute(mu - test_label.numpy())
    mse_error = np.square(mu - test_label.cpu().numpy().flatten())
    error = np.sqrt(mse_error.mean())

    test_label = test_label.cpu().numpy().flatten()
    mu_up = mu + std
    mu_down = mu - std
    smaller_than_up = (test_label < mu_up)
    larger_than_down = (test_label > mu_down)

    ratio = ((smaller_than_up & larger_than_down) * 1.0).sum() / len(test_label)

    return error, std.mean(), ratio, log_ll.mean().detach().cpu().numpy()



def main():
    global opt
    train_data, train_label, valid_data, valid_label,_= get_dataset(dataset_name)
    train_data, train_label, valid_data, valid_label = [item.to(device) for item in \
                                                        [train_data, train_label, valid_data, valid_label]]
    
    
    setup_seed(opt.seed)

    if model_name == 'dilgp':
        gp = DILGP(opt.envlr, opt.eistep, opt.lambdae, opt.usekmeans,
                opt.length_scale, opt.noise_scale, opt.amplitude_scale).to(device)
    elif model_name == 'gp_kernel':

        kernel = eval(opt.kernel)()
        
        gp = DILGPKernel(kernel, opt.envlr, opt.eistep, opt.lambdae, opt.usekmeans).to(device)
    else:
        gp = GP(opt.length_scale, opt.noise_scale, opt.amplitude_scale).to(device)
    
    
    optimizer = SGD(gp.parameters(), lr=opt.gplr)

    gp.fit(train_data, train_label)

    for i in tqdm(range(opt.epoch)):
        d_train = gp.train_step(train_data, train_label, optimizer)

    with torch.no_grad():
        test_error, test_std, ratio, ll = test(gp, valid_data, valid_label)

    print('Model: gp with',opt.kernel)
    print('RMSE: ',test_error)
    print('Coverage Rate: ', ratio)   
        


if __name__ == "__main__":
    main()

