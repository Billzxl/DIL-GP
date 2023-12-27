import argparse
import numpy as np
import random
import torch
import os
from DILGP import DILGP
from gp import GP
from torch.optim import SGD
# from get_my_data import get_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.distributions.normal import Normal
from matplotlib.transforms import Bbox

parser = argparse.ArgumentParser(description='DILGP Tuning')

parser.add_argument('--model_name',
                    help='choose model, gp or dilgp',
                    default='gp',
                    choices= ['gp','dilgp'],
                    type=str)

parser.add_argument('--dataset_name',
                    help='choose dataset',
                    default='synthetic',
                    choices= ['synthetic'],
                    type=str)

parser.add_argument('--envlr',
                    help='learning rate for env_w',
                    default=0.001,
                    type=float)
parser.add_argument('--gplr',
                    help='learning rate for gp parameters',
                    default=0.005,
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
    # print(train_data.shape,train_label.shape,valid_data.shape,valid_label.shape)
    # assert 0
    return train_data, train_label, valid_data, valid_label


def test(gp, test_data, test_label):
    mu, var = gp.forward(test_data)

    var = torch.clamp(var, min=1e-10, max=1e10)
    std = torch.sqrt(var)

    mu = mu.detach().numpy().flatten()
    std = torch.sqrt(var).detach().numpy().flatten()
    
    mse_error = np.square(mu - test_label.numpy().flatten())
    error = np.sqrt(mse_error.mean())

    test_label = test_label.numpy().flatten()
    mu_up = mu + std
    mu_down = mu - std
    smaller_than_up = (test_label < mu_up)
    larger_than_down = (test_label > mu_down)

    ratio = ((smaller_than_up & larger_than_down) * 1.0).sum() / len(test_label)

    return error,  ratio



def plot(gp, train_data, train_label, test_data, test_label, error):
    
    if not os.path.exists('./debug/'):
        os.makedirs('./debug/')
    
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
    
    plt.plot(grid.flatten(), mu, color='purple', linewidth=2,alpha=0.8)
    gt1 =  (torch.sin(grid1 * 2 * np.pi /4).flatten()*3).numpy().flatten()
    gt2 = (-torch.sin((grid2-6.5) * 2 * np.pi / 64).flatten() + 0.5).numpy().flatten()

    plt.fill_between(grid.flatten(), y1=mu+std, y2=mu-std, alpha=0.5, color='mediumslateblue')

    plt.scatter(test_data.flatten(), test_label, 
             c='orange', s=15)
    plt.scatter(train_data.flatten(), train_label, 
                c='blue', s=15)

    plt.xticks([])
    plt.yticks([])

    if opt.model_name == 'dilgp':
        plt.text(2.5,3,f'DIL-GP, RMSE={error:.4f}')
        # plt.title(f'DIL-GP, RMSE={error:.4f}')
        plt.savefig(f'result/dilgp_result.png', bbox_inches=Bbox.from_bounds(0.99, 0.31, 6.23, 2.34))

    else:
        # plt.title(f'GP, RMSE={error:.4f}')
        plt.text(3.5,3,f'GP, RMSE={error:.4f}')
        plt.savefig(f'result/gp_result.png',
                    bbox_inches=Bbox.from_bounds(0.99, 0.31, 6.23, 2.34))
              
    plt.cla()  
    
    plt.subplots(figsize=(8, 1))
    plt.ylim(-4, 3)

    plt.plot(grid1.flatten(), mu1-gt1, color='purple', linewidth=2,alpha=1)
    plt.plot(grid2.flatten(), mu2-gt2, color='purple', linewidth=2,alpha=1)
    plt.fill_between(grid1.flatten(), y1=mu1-gt1, y2=0, alpha=0.5, color='mediumslateblue')
    plt.fill_between(grid2.flatten(), y1=mu2-gt2, y2=0, alpha=0.5, color='mediumslateblue')
    plt.xticks([])
    plt.yticks([])




def main():
    global opt

    train_data, train_label, valid_data, valid_label = get_dataset(opt.dataset_name,0)
    setup_seed(opt.seed)

    if opt.model_name == 'dilgp':
        gp = DILGP(opt.envlr, opt.eistep, opt.lambdae, opt.usekmeans,
                opt.length_scale, opt.noise_scale, opt.amplitude_scale)#.cuda()
    else:
        gp = GP(opt.length_scale, opt.noise_scale, opt.amplitude_scale)

    optimizer = SGD(gp.parameters(), lr=opt.gplr)

    gp.fit(train_data, train_label)

    for i in tqdm(range(opt.epoch)):
        gp.train_step(train_data, train_label, optimizer)

    with torch.no_grad():
        test_error, ratio = test(gp, valid_data, valid_label)

    plot(gp, train_data, train_label, valid_data, valid_label, test_error)

    print('Model: ',opt.model_name)
    print('RMSE: ',test_error)
    print('Coverage Rate: ', ratio)


if __name__ == "__main__":
    main()

