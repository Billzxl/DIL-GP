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
# from get_my_data import get_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from torch.distributions.normal import Normal
from matplotlib.transforms import Bbox

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
                    default=1,
                    type=int)
parser.add_argument('--lambdae',
                    help='lambda coefficient, loss = marginal_likelihood + npenalty * lambda',
                    default=1,
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
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

# setup_seed(0)
dataset_name = 'synthetic'
model_name = 'oodgp'
# model_name = 'gp'

def get_dataset(dataset_name, s):
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



def test_plot(gp, train_data, train_label, test_data, test_label, error):
    grid = torch.linspace(-3, 8.5, 200)[:,None]
    grid1 = torch.linspace(-3, 3.5, 200)[:,None]
    grid2 = torch.linspace(3.5, 8.5, 200)[:,None]
    grid3 = torch.linspace(5.5, 8.5, 200)[:,None]
    test_data = np.clip(test_data, a_min=-3, a_max=8)

    mu, var = gp.forward(grid)
    mu1,_ =  gp.forward(grid1)
    mu2,_ =  gp.forward(grid2)
    mu3,_ =  gp.forward(grid3)
    mu = mu.detach().numpy().flatten()
    mu1 = mu1.detach().numpy().flatten()
    mu2= mu2.detach().numpy().flatten()
    mu3= mu3.detach().numpy().flatten()
    std = torch.sqrt(var).detach().numpy().flatten()

    plt.rcParams.update({'font.size': 20})
    plt.subplots(figsize=(8, 3))
    plt.ylim(-4, 4)
    # plt.axvline(x=3.5, color='grey', linestyle='--',linewidth = 2)
    plt.plot(grid.flatten(), mu, color='purple', linewidth=2,alpha=0.8)
    gt1 =  (torch.sin(grid1 * 2 * np.pi /4).flatten()*3).numpy().flatten()
    gt2 = (-torch.sin((grid2-6.5) * 2 * np.pi / 64).flatten() + 0.5).numpy().flatten()
    gt3 = (-torch.sin((grid3-6.5) * 2 * np.pi / 64).flatten() + 0.5).numpy().flatten()
    # plt.plot(grid1.flatten(), gt1, color='black', linewidth=1,alpha=1)
    # plt.plot(grid2.flatten(), gt2, color='black', linewidth=1,alpha=1)
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
    # plt.text(2.5,3,f'DIL-GP, RMSE={error:.4f}')
   
    # plt.tight_layout()
    # plt.tight_layout()
    if model_name == 'oodgp':
        plt.text(2.5,3,f'DIL-GP, RMSE={error:.4f}')
        # plt.title(f'DIL-GP, RMSE={error:.4f}')
        plt.savefig(f'debug/6_train_oodgp_result.png', bbox_inches=Bbox.from_bounds(0.99, 0.31, 6.23, 2.34))
        #Bbox.from_bounds(0.5, 0, 6, 4.7))
        # plt.title(f'IRL-GP, error={error:.4f}')
    else:
        # plt.title(f'GP, RMSE={error:.4f}')
        plt.text(3.5,3,f'GP, RMSE={error:.4f}')
        plt.savefig(f'debug/6_train_gp_result.png',
                    bbox_inches=Bbox.from_bounds(0.99, 0.31, 6.23, 2.34))
        # plt.title(f'GP, error={error:.4f}')
        
        
        
        
        
        
        
        
        
        
        
        
    plt.cla()  
    
    plt.subplots(figsize=(8, 1))
    plt.ylim(-4, 3)
    # print(type(mu))
    # print(type(gt1))
    plt.plot(grid1.flatten(), mu1-gt1, color='purple', linewidth=2,alpha=1)
    plt.plot(grid2.flatten(), mu2-gt2, color='purple', linewidth=2,alpha=1)
    plt.fill_between(grid1.flatten(), y1=mu1-gt1, y2=0, alpha=0.5, color='mediumslateblue')
    plt.fill_between(grid2.flatten(), y1=mu2-gt2, y2=0, alpha=0.5, color='mediumslateblue')
    plt.xticks([])
    plt.yticks([])
    # plt.axvline(x=3.5, color='grey', linestyle='--',linewidth = 2)
    # plt.tight_layout()
    if model_name == 'oodgp':
    
        # plt.title(f'DIL-GP, RMSE={error:.4f}')
        plt.savefig(f'debug/3_train_oodgp_result_cp.png', bbox_inches=Bbox.from_bounds(0.99, 0.09, 6.23, 0.8))
        #Bbox.from_bounds(0.5, 0, 6, 4.7))
        # plt.title(f'IRL-GP, error={error:.4f}')
    else:
        # plt.title(f'GP, RMSE={error:.4f}')
     
        plt.savefig(f'debug/3_train_gp_result_cp.png',
                    bbox_inches=Bbox.from_bounds(0.99, 0.09, 6.23, 0.8))
        # plt.title(f'GP, error={error:.4f}')     
        
        
        
        
    plt.cla()  
    
    plt.subplots(figsize=(4, 1))
    plt.ylim(-0.3, 0.3)
    # print(type(mu))
    # print(type(gt1))
    # plt.plot(grid1.flatten(), mu1-gt1, color='purple', linewidth=2,alpha=1)
    plt.plot(grid3.flatten(), mu3-gt3, color='purple', linewidth=2,alpha=1)
    # plt.fill_between(grid1.flatten(), y1=mu1-gt1, y2=0, alpha=0.5, color='mediumslateblue')
    plt.fill_between(grid3.flatten(), y1=mu3-gt3, y2=0, alpha=0.5, color='mediumslateblue')
    plt.xticks([])
    plt.yticks([])
    # plt.axvline(x=3.5, color='grey', linestyle='--',linewidth = 2)
    # plt.tight_layout()
    if model_name == 'oodgp':
    
        # plt.title(f'DIL-GP, RMSE={error:.4f}')
        plt.savefig(f'debug/3_train_oodgp_result_cp_part.png', bbox_inches=Bbox.from_bounds(0.48, 0.09, 3.15, 0.8))
        #Bbox.from_bounds(0.5, 0, 6, 4.7))
        # plt.title(f'IRL-GP, error={error:.4f}')
    else:
        # plt.title(f'GP, RMSE={error:.4f}')
     
        plt.savefig(f'debug/3_train_gp_result_cp_part.png',
                    bbox_inches=Bbox.from_bounds(0.48, 0.09, 3.15, 0.8))
        # plt.title(f'GP, error={error:.4f}')
        
    plt.cla()  
    
    
    
    
    
    
    
    

    if model_name == 'oodgp':
        env_train = gp.env_w.flatten()
        env1 = (env_train.sigmoid() < 0.5)
        env2 = (env_train.sigmoid() >= 0.5)
        plt.plot(train_data.flatten()[env1], train_label[env1], '.', color='blue')
        plt.plot(train_data.flatten()[env2], train_label[env2], '.', color='yellow')

        plt.savefig(f'debug/train_oodgp_env.png')

        plt.cla()


def main():
    global opt

    error_diff_seed = []
    ratio_diff_seed = []
    ll_diff_seed = []
    std_diff_seed = []
    error0 = None
    ratio0 = None
    ll0 = None
    std0 = None
    # for s in range(10):
    gplrs = [0.01, 0.003, 0.001, 0.0003, 0.0001] # exp in paper
    # gplrs = [0.0008,0.02, 0.003,0.005, 0.001,0.002, 0.0003,0.0006, 0.0001,0.0002]
    # gplrs = [0.001]
    for i, gplr in enumerate(gplrs):
        opt.gplr = gplr
        train_data, train_label, valid_data, valid_label = get_dataset('syn',0)
        setup_seed(0)
        # ONLY for regression !
        # TODO： add classification
        if model_name == 'oodgp':
            gp = OODGP(opt.envlr, opt.eistep, opt.lambdae, opt.usekmeans,
                    opt.length_scale, opt.noise_scale, opt.amplitude_scale)#.cuda()
        else:
            gp = GP(opt.length_scale, opt.noise_scale, opt.amplitude_scale)

        optimizer = SGD(gp.parameters(), lr=opt.gplr)

        gp.fit(train_data, train_label)

        l_error = []; l_std = []
        test_error, test_std, _, _ = test(gp, valid_data, valid_label)
        l_error.append(test_error.mean())
        l_std.append(test_std.mean())

        l_loss = []; l_length = []; l_noise = []; l_amp = [];
        for i in tqdm(range(opt.epoch)):
            d_train = gp.train_step(train_data, train_label, optimizer)
            l_loss.append(d_train['loss'])
            l_length.append(d_train['length'])
            l_noise.append(d_train['noise'])
            l_amp.append(d_train['amplitude'])
            # print(d_train['loss'])
            with torch.no_grad():
                test_error, test_std, ratio, ll = test(gp, valid_data, valid_label)
                l_error.append(test_error.mean())
                l_std.append(test_std.mean())

        # print('l_error', l_error) # TODO: check other standards?
        error = l_error[-1]
        test_plot(gp, train_data, train_label, valid_data, valid_label, error)

        error_diff_seed.append(error)
        ratio_diff_seed.append(ratio)
        ll_diff_seed.append(ll)
        std_diff_seed.append(test_std)
        print(error, ratio, ll, test_std)

        if np.abs(gplr - 0.001) < 1e-10:
            error0  = error
            ratio0 = ratio
            ll0 = ll
            std0 = test_std

    print(error_diff_seed)
    diff = np.abs(np.array(error_diff_seed) - error0)
    print('mean_error,max d')
    print(error0, diff.max())

    diff = np.abs(np.array(ratio_diff_seed) - ratio0)
    print(ratio0, diff.max())

    # diff = np.abs(np.array(ll_diff_seed) - ll0)
    # print(ll0, diff.max())

    # diff = np.abs(np.array(std_diff_seed) - std0)
    # print(std0, diff.max())




if __name__ == "__main__":
    main()

