import argparse

import sklearn
import sklearn.datasets as sk_data
import numpy as np
import random
import torch
import os
from OODGP import OODGP
from gp import GP
from OODGPKernel import OODGPKernel
from torch.optim import SGD
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kernels import *
from get_my_data import get_dataset
# from get_housing import get_dataset

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

parser = argparse.ArgumentParser(description='OODGP Tuning')
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
                    default=0.000000000000000000000000000000000030,
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
dataset_name = "auto_mobile"
# dataset_name = "housing"

# gp: 0.6249123
# gp irm: 0.86464185
# oodgp: 0.8157944

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
    
    error_diff_seed = []
    ratio_diff_seed = []
    ll_diff_seed = []
    std_diff_seed = []
    error0 = None
    ratio0 = None
    ll0 = None
    std0 = None
    gplrs = [0.01, 0.006, 0.003, 0.001]
    for i, gplr in enumerate(gplrs):
        opt.gplr = gplr
        setup_seed(opt.seed)
        # ONLY for regression !
        # TODO： add classification
        if model_name == 'oodgp':
            gp = OODGP(opt.envlr, opt.eistep, opt.lambdae, opt.usekmeans,
                    opt.length_scale, opt.noise_scale, opt.amplitude_scale).to(device)
        elif model_name == 'gp_kernel':
            # kernel = RationalQuadraticKernel()
            kernel = DotProductKernel()
            
            gp = OODGPKernel(kernel, opt.envlr, opt.eistep, opt.lambdae, opt.usekmeans).to(device)
        else:
            gp = GP(opt.length_scale, opt.noise_scale, opt.amplitude_scale).to(device)
        
        optimizer = torch.optim.Adam(gp.parameters(), lr=opt.gplr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=100)

        gp.fit(train_data, train_label)
        
        l_error = []; l_std = []
        l_ratio = []; l_ll = []
        for e in tqdm(range(opt.epoch)):
            d_train = gp.train_step(train_data, train_label, optimizer)
            with torch.no_grad():
                test_error, test_std, ratio, ll = test(gp, valid_data, valid_label)
                l_error.append(test_error.mean())
                l_std.append(test_std.mean())
                l_ratio.append(ratio)
                l_ll.append(ll)

        error_idx = np.argmin(np.array(l_error))
        error = l_error[error_idx]
        ratio = l_ratio[error_idx]
        ll = l_ll[error_idx]
        test_std = l_std[error_idx]
        print(error)

        error_diff_seed.append(error)
        ratio_diff_seed.append(ratio)
        ll_diff_seed.append(ll)
        std_diff_seed.append(test_std)
        print(error, ratio, ll, test_std)

        if np.abs(gplr - 0.01) < 1e-10:
            error0  = error
            ratio0 = ratio
            ll0 = ll
            std0 = test_std

    print(error_diff_seed)
    diff = np.abs(np.array(error_diff_seed) - error0)
    print('error', error0, diff.max())

    diff = np.abs(np.array(ratio_diff_seed) - ratio0)
    print('ratio', ratio0, diff.max())

    diff = np.abs(np.array(ll_diff_seed) - ll0)
    print('ll', ll0, diff.max())

    diff = np.abs(np.array(std_diff_seed) - std0)
    print('std', std0, diff.max())
    
    
    
    
    
    
    
    

    # # gp.initialize(train_data, train_label, optimizer)

    # # optimizer = torch.optim.Adam(gp.parameters(), lr=opt.gplr)

    # l_error = []; l_std = []
    # test_error, test_std = test(gp, valid_data, valid_label)
    # l_error.append(test_error.mean())
    # l_std.append(test_std.mean())

    # l_loss = []; l_length = []; l_noise = []; l_amp = []; l_loss2 = []
    # # for i in tqdm(range(opt.epoch)):
    # for i in range(opt.epoch):
    #     d_train = gp.train_step(train_data, train_label, optimizer)
    #     l_loss.append(d_train['loss'])
    #     # l_loss2.append(gp.marginal_likelihood.cpu())
    #     # l_length.append(d_train['length'])
    #     # l_noise.append(d_train['noise'])
    #     # l_amp.append(d_train['amplitude'])
    #     for i in range(1,6):
    #         _, _, valid_data, valid_label = get_dataset(dataset_name,1)
        
        
    #         with torch.no_grad():
    #             test_error, test_std = test(gp, valid_data, valid_label)
    #             l_error.append(test_error.mean())
    #             l_std.append(test_std.mean())

    #         scheduler.step()
    #         if i % 1 == 0:
    #             print('error', l_error[-1])

    # # print('l_error', l_error) # TODO: check other standards?
    # error = l_error[-1]

    # print(opt, error)




if __name__ == "__main__":
    main()

