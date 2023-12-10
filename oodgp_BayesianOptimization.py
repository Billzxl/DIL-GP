# example of bayesian optimization for a 1d function from scratch
from math import sin
from math import pi
import numpy as np
import time
import os
from numpy import arange
from numpy import vstack
from numpy import argmax,argmin
from numpy import asarray
from numpy.random import normal
# from numpy.random import random
import random
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from warnings import catch_warnings
from warnings import simplefilter
from matplotlib import pyplot
import torch
from torch.optim import SGD,Adam
from OODGP_bo import OODGP
from OODGPKernel_bo import OODGPKernel
from gp import GP
from sklearn.preprocessing import StandardScaler

from io import StringIO
import contextlib
import sys
from tqdm import tqdm
from kernels_bo import RationalQuadraticKernel, DotProductKernel, MaternKernel25
# from zxl.ml.prj.from_git.OODGP.kernels import RationalQuadraticKernel
sys.path.append('PID/meta_test/quadsim')
from PID.meta_test.quadsim.func_experiment import calculate_ACE
from tqdm import trange
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import argparse
from scipy.optimize import minimize

from functools import partial
parser = argparse.ArgumentParser(description='OODGP BO')

parser.add_argument('--kernel',
                    help='hover, fig-8,sin_forward,spiral_up',  #['hover', 'fig-8','sin_forward','spiral_up']
                    default='MaternKernel25',
                    type=str)

parser.add_argument('--t_name',
                    help='hover, fig-8,sin_forward,spiral_up',  #['hover', 'fig-8','sin_forward','spiral_up']
                    default=None,
                    type=str)

parser.add_argument('--batchsize',
                    help='batchsize',  #['hover', 'fig-8','sin_forward','spiral_up']
                    default=10000,#10000
                    type=int)

parser.add_argument('--ucb_k',
                    help='k value of ucb sample', 
                    default=2.5,#尝试 [1,3,5,0.5]
                    type=float)

parser.add_argument('--lr1',
                    help='lr outside',
                    default=0.1,#[0.1,0.01,0.001] 0.1模型可能异常终止，若发生记录"异常终止"即可
                    type=float)

parser.add_argument('--lambdae',
                    help='lambdae',
                    default=0.1,#[0.1,1e-5]
                    type=float)

parser.add_argument('--device',
                    help='device', #机器有什么就用什么
                    default=0,
                    type=int)

parser.add_argument('--warm_up_num',
                    help='warm_up_num', #机
                    default=100000,#100000
                    type=int)


parser.add_argument('--try_acq_num',
                    help='try_acq_num', #
                    default=250,#250
                    type=int)

parser.add_argument('--save_name',
                    help='save one reult with one name',
                    default=None,#不调
                    type=str)


parser.add_argument('--envlr',
                    help='lr inside',
                    default=0.001,#不调
                    type=float)

parser.add_argument('--seed',
                    help='random seed',#不调
                    default=0,
                    type=int)


parser.add_argument('--num1',
                    help='initial size = num1**3', #不调
                    default=2,
                    type=int)

parser.add_argument('--num2',
                    help='seach size = num2**3', #不调
                    default=20,
                    type=int)

parser.add_argument('--epoch',
                    help='seach size = num1**3', #不调
                    default=100,
                    type=int)

parser.add_argument('--model',
                    help='oodgp,gp,rf,mlp', #不调
                    default='oodgp',
                    type=str)


opt = parser.parse_args()
if(opt.save_name == None):
    opt.save_name = 'ucb_k_'+str(opt.ucb_k)+'_lr1_' + str(opt.lr1)+'_lambdae_' + str(opt.lambdae)+'_num1_'+str(opt.num1)+'_kernel_'+str(opt.kernel)+'_epoch_'+str(opt.epoch)+'_newacq_D_seed_'+str(opt.seed)+'_0516'

print(opt)
print(opt.save_name)

opt.lower = 0.0001
opt.upper = 0.1

#default值
# num1 = opt.num1
# num2 = opt.num2
# epoch = opt.epoch



# ace_score = calculate_ACE(update_pid=update_pid_template, need_pdf=False)

device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")

if opt.model in ['rf','mlp']:
    device = torch.device("cpu")
# device = torch.device('cpu')
# hyper_name = "MC_PITCHRATE_P"


update_pid_template = {
    "MC_ROLLRATE_P": 0.4,
    "MC_PITCHRATE_P": 0.4,
    "MC_YAWRATE_P": 0.1,
    "MC_ROLLRATE_I": 0.07,
    "MC_PITCHRATE_I": 0.07,
    "MC_YAWRATE_I": 0.0005,
    "MC_ROLLRATE_D": 0.0016,
    "MC_PITCHRATE_D": 0.0016,
    "MC_YAWRATE_D": 0.01
}
   
# opt_hyper_list = ["MC_ROLLRATE_P","MC_PITCHRATE_P", "MC_YAWRATE_P"]
opt_hyper_list = ["MC_ROLLRATE_D","MC_PITCHRATE_D", "MC_YAWRATE_D"]
   
q_kwargs = {
    'Vwind' : np.array((0.0, 0, 0)),            # mean wind speed
    'wind_model': 'iid-uniform',                # {'iid', 'random-walk'}
    'Vwind_cov' : 0.,                        # how quickly the wind changes
    # 'wind_constraint' : 'hard',               # 'hard' wind constraint limits wind speed to be within Vwind_gust of the mean speed
    'Vwind_gust' : np.array((5.0, 0., 2.5)),    # for hard wind constrant, wind speed is in the range Vwind +/- Vwind_gust
    'wind_update_period' : 2.0,                 # seconds between wind speed changes
    't_stop' : 15.,
}

hyper_range = {
    
    "MC_ROLLRATE_P": [0.1,5],
    "MC_PITCHRATE_P": [0.1,5],
    "MC_YAWRATE_P": [0.1,5]   ,
    "MC_ROLLRATE_D": [0.00001,0.1],
    "MC_PITCHRATE_D": [0.00001,0.1],
    "MC_YAWRATE_D": [0.00001,0.1]   
}
   
def objective1(X,env,record,name,test_name):
    assert env in [0,1]
    buffer = StringIO()
    X = X.detach().cpu().numpy()
    # update_pid_template["MC_ROLLRATE_P"] = X[0]
    # update_pid_template["MC_PITCHRATE_P"] = X[1]
    # update_pid_template["MC_YAWRATE_P"] = X[2]
    
    update_pid_template["MC_ROLLRATE_D"] = X[0]
    update_pid_template["MC_PITCHRATE_D"] = X[1]
    update_pid_template["MC_YAWRATE_D"] = X[2]
    
    
    
    
    if env == 0:
        q_kwargs['Vwind'] = np.array((0.0, 0., 0.))
        # q_kwargs['Vwind_cov'] = 1.0
        q_kwargs['Vwind_gust'] = np.array((5.0, 0., 2.5))
    
    else:
        q_kwargs['Vwind'] = np.array((3.0, 0., 1.0))
        # q_kwargs['Vwind_cov'] = 7.5
        q_kwargs['Vwind_gust'] = np.array((2.0, 0., 1.0))
        
    with contextlib.redirect_stdout(buffer):
        return calculate_ACE(update_pid=update_pid_template, q_kwargs = q_kwargs, Name=name,test_name=test_name,record = record,opt=opt,need_pdf=False)

def objective(x, env,record,name,test_name):
    x = x.detach().cpu().numpy()
    # noise = normal(loc=0, scale=noise)
    
    # print(type(sin(5 * np.pi * x)**6.0 * x.item()**2))
    # print(sin(5 * np.pi * x)**6.0 * x.item()**2)
    # print(type((sin(np.pi * x)**6.0)))
    # print((sin(np.pi * x)**6.0))
    
    
    # return (sin(5 * np.pi * x)**6.0 * x.item()**2) 
    # return ( -(x-0.5)**2) 
    return (sin(5*np.pi * x[0])**6.0)

def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)

def generate_env():
    # num = random.random()
    # random.seed(int(time.time()))
    num = random.random()
    # print(num)
    if num < 0.7:
        return 0
    else:
        return 1

#用拟合好的模型预测黑盒输出
def surrogate(model, X,model_name):
    # print(type(X))
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X).to(device)
    if len(X.shape) == 1:
        X = X.unsqueeze(1).T
    # print('Xshape',X.shape)
    batch_size = opt.batchsize
    
    if(model_name in ['gp','oodgp']):
        y_mu = torch.tensor(np.zeros((X.shape[0],1))).to(device)
        y_std = torch.tensor(np.zeros((X.shape[0],1))).to(device)
        # print('ymu ',y_mu.shape)
        
        # print(len(X),batch_size)
        
        
        for i in range(0, len(X), batch_size):
            # print('i,i+bt ',i,i+batch_size)
            x_batch = X[i:i+batch_size]
            # print(x_batch)
            y_batch1 , y_batch2 = model.forward(x_batch)
            # print(y_batch1.shape)
            # print(y_batch2.unsqueeze(1).shape)
            y_mu[i:i+batch_size] = y_batch1
            y_std[i:i+batch_size] = y_batch2.unsqueeze(1)
            
            
        # print('ymu',y_mu)
        # print('y_std',y_std)
        return y_mu.detach().cpu(), y_std.detach().cpu()
    
    
    
    elif(model_name in ['mlp','rf']):
        return model.predict(X.detach().cpu()),0
        
        
        
#在候选集合中进行选择,此值越小越应该选
def acquisition(Xsamples, model,model_name):
    # print('m name',model_name)
    if model_name in ['rf','mlp']:
        mu,_ = surrogate(model, Xsamples,model_name)
        return mu
    
    elif(model_name in ['oodgp','gp']):
        # calculate the best surrogate score found so far
        # yhat, _ = surrogate(model, X,model_name)
        # best = max(yhat)
        # calculate mean and stdev via surrogate function
        # print(Xsamples.shape)
        mu, var = surrogate(model, Xsamples,model_name)
        # print(mu.shape)
        # print(var.shape)
        
        
        std = torch.sqrt(var).detach().cpu().numpy().flatten()
        # print('mu shape',mu.shape)
        mu = mu.detach().cpu().numpy().flatten()
        # print('mu shape',mu.shape)
        # calculate the probability of improvement
        # probs = norm.cdf((mu - best.detach().cpu().numpy()) / (std+1E-9))
        # print(f'mu: {mu},std: {std}')
        probs = mu - std * opt.ucb_k
        
        #max/min ?

        return(probs)

    #print(probs)
    
    # mu = surrogate(model, Xsamples,model_name)
    # return mu

# optimize the acquisition function
def opt_acquisition(X, y, model, scale_list, scaler_list,model_name):
    # random search, generate random samples
    # Xsamples = torch.tensor(np.random.random(100)*(scale[1] - scale[0])+scale[0]).to(device)
    
    # Xsamples = Xsamples.reshape(len(Xsamples), 1)
    
    # num = opt.num2
    
    # x1 = np.random.random(num)*(scale_list[0][1] - scale_list[0][0])+scale_list[0][0]
    # x2 = np.random.random(num)*(scale_list[1][1] - scale_list[1][0])+scale_list[1][0]
    # x3 = np.random.random(num)*(scale_list[2][1] - scale_list[2][0])+scale_list[2][0]

    # print(type(opt.warm_up_num))

    x_tries = np.random.uniform(low=opt.lower, high=opt.upper, size=(opt.warm_up_num, 3))

    
    
    bounds = np.zeros((3,2))
    
    
    for i in range(x_tries.shape[1]):
        scaler = scaler_list[i]
        x_tries[:,i] = (scaler.transform(x_tries[:,i].reshape(-1,1))).flatten()
        bounds[i] = scaler.transform(np.array([opt.lower,opt.upper]).reshape(-1,1)).flatten()
        
    x_tries = torch.tensor(x_tries).to(device)
    
    # print(bounds)
    
    # print(x_tries.shape)
    
    acquisition_1 = partial(acquisition, model = model, model_name = model_name)
    # print('start trying 1')
    ys = acquisition_1(x_tries)
    # print(ys.shape)
    x_min = x_tries[ys.argmin()]
    min_acq = ys.min()#随机找到的点中最好的


    # Explore the parameter space more thoroughly
    x_seeds = np.random.uniform(low=opt.lower, high=opt.upper,
                                   size=(opt.try_acq_num, 3))


    for i in range(x_seeds.shape[1]):
        scaler = scaler_list[i]
        x_seeds[:,i] = (scaler.transform(x_seeds[:,i].reshape(-1,1))).flatten()
    # print('start trying 2')
    for x_try in x_seeds:
        # print(x_try.shape)
        # Find the minimum of minus the acquisition function
        res = minimize(acquisition_1,
                       x_try,
                       bounds=bounds,
                       method="L-BFGS-B")

        # See if success
        if not res.success:
            continue

        # Store it if better than previous minimum(maximum).
        if min_acq is None or np.squeeze(res.fun) <= min_acq:
            x_min = res.x
            min_acq = np.squeeze(res.fun)

    # Clip output to make sure it lies within the bounds. Due to floating
    # point technicalities this is not always the case.
    # print(bounds[0][0],bounds[0][1])
    # print('finish trying 1,2')
    # print('xxx')
    # print(type(x_min))
    if  isinstance(x_min, torch.Tensor):
        x_min = x_min.detach().cpu().numpy()
    # print(type(x_min))
    return torch.tensor(np.clip(x_min, bounds[0][0], bounds[0][1])).to(device)








    # # 生成三个维度的坐标
    # xv, yv, zv = np.meshgrid(x1, x2, x3, indexing='ij')

    # # 将三个维度的坐标组成一个 (n**3, 3) 的 Tensor，然后随机选择 m 个采样点
    # # indices = np.random.choice(n**3, size=m, replace=False)
    # Xsamples = torch.tensor(np.stack([xv.ravel(), yv.ravel(), zv.ravel()], axis=-1))
    
    # # print('Xsamples before scaler',Xsamples)
    
    
    
    # # print('Xsamples shape ',Xsamples.shape)
    # for i in range(Xsamples.shape[1]):
    #     scaler = scaler_list[i]
    #     Xsamples[:,i] = torch.tensor(scaler.transform(Xsamples[:,i].reshape(-1,1))).squeeze()
    
    # Xsamples = Xsamples.to(device)
    # # print('Xsamples after scaler',Xsamples)
    # # Xsamples = torch.tensor(scaler.transform(Xsamples.cpu())).to(device)
    
    
    # # print('Xsamples',Xsamples)
    # # calculate the acquisition function for each sample
    # scores = acquisition(X, Xsamples, model,model_name)
    # # print('scores.shape',scores.shape)

    # # # locate the index of the largest scores
    # ix = argmin(scores)
    # # print('ix',ix)
    # return Xsamples[ix]







# plot real observations vs surrogate function
def plot(X, y, env,model,name,scaler1,scaler2):
    # print(env)

    X_ori = scaler1.inverse_transform(X.cpu())
    y_ori = scaler2.inverse_transform(y.cpu())

    
    colors = ['#FFA500','#006400']
    color_list = [colors[x] for x in env]
    # print('env: ',env)
    # print('colors: ',color_list)

    # scatter plot of inputs and real objective function
    # pyplot.scatter(X.detach().cpu(), y.detach().cpu(),c = '#006400',zorder = 20)
    pyplot.scatter(X_ori, y_ori,c = color_list,zorder = 20,alpha=0.3)
    # line plot of surrogate function across domain
    Xsamples = torch.tensor(np.linspace(start=min(X.cpu()), stop=max(X.cpu()), num=1000)).to(device)
    
    Xsamples = Xsamples.reshape(len(Xsamples), 1)
    ysamples, _ = surrogate(model, Xsamples,model_name)
    
    Xsamples_ori = scaler1.inverse_transform(Xsamples.detach().cpu())
    ysamples_ori = scaler2.inverse_transform(ysamples.detach().cpu())
    
    # print(Xsamples)
    # print(ysamples)
    pyplot.plot(Xsamples_ori, ysamples_ori,zorder = 30)
    # pyplot.plot(Xsamples.detach().cpu(), ysamples.detach().cpu(),zorder = 30)
    # show the plot
    pyplot.title(name)
    pyplot.savefig('fig/PID/4_'+name,dpi = 300)
    pyplot.clf()



# def scale_x(X,scaler_list):
    
def inverse_scale_x(X,scaler_list):
    
    inverse_x = torch.zeros([X.shape[0],X.shape[1]])
    
    for i in range(len(scaler_list)):
        scaler = scaler_list[i]
        inverse_x[:,i] = torch.tensor(scaler.inverse_transform(X[:,i].reshape(-1,1))).squeeze()
    return inverse_x


def BayesianOptimization(model,name,test_name):
    seed = opt.seed
    setup_seed(seed)
    scale_list = []
    print(f'start {name} {test_name}')
    
    for hyper_name in opt_hyper_list:
        scale = hyper_range[hyper_name]
        scale_list.append(scale)
    
    # sample the domain sparsely with noise
    # X = torch.rand(20,1)*(scale[1] - scale[0]) + scale[0]
    num = opt.num1
    
    x1 = np.linspace(start = scale_list[0][0], stop = scale_list[0][1], num=num)
    # print('x1: ',x1)
    # assert 0
    x2 = np.linspace(start = scale_list[1][0], stop = scale_list[1][1], num=num)
    x3 = np.linspace(start = scale_list[2][0], stop = scale_list[2][1], num=num)

    # 生成三个维度的坐标
    xv, yv, zv = np.meshgrid(x1, x2, x3, indexing='ij')

    # 将三个维度的坐标组成一个 (n**3, 3) 的 Tensor，然后随机选择 m 个采样点
    # indices = np.random.choice(n**3, size=m, replace=False)
    X = torch.tensor(np.stack([xv.ravel(), yv.ravel(), zv.ravel()], axis=-1))

    # print(X.shape)
    # print(X)
    # assert 0
    
    # for i in len(opt_hyper_list):
    #     x[i] = torch.tensor(np.linspace(start = scale_list[i][0], stop = scale_list[i][1], num=5).reshape(-1,1))
    #     X = torch.tensor(np.linspace(start = scale[0], stop = scale[1], num=5).reshape(-1,1))
    
    print('start calculating origin scores')
    
    # for i in range(10):
    #     env = generate_env()
    #     print(env)
    # assert 0
    
    
    y = []
    env_list = []
    for x in tqdm(X):
        env = generate_env()
        # print(env)
        y.append(objective(x,env,record=0,name=name,test_name=test_name))
        env_list.append(env)
    y = asarray(y)
    print('finish calculating origin scores')

    # reshape into rows and cols
    # X = X.reshape(len(X), 1)
    y = y.reshape(len(y), 1)

    # print(X.shape)
    # print(y.shape)
    # assert 0

    # oodgp = OODGP().to(device)
    # opt2 = SGD(oodgp.parameters(), lr=0.001, momentum = 0.1) 
    if name in ['gp','oodgp']:
        opt2 = SGD(model.parameters(), lr = opt.lr1, momentum = 0)       



    # print('X before scaler',X)
    scaler_list = []
    for i in range(X.shape[1]):
        scaler = StandardScaler()
        X[:,i] = torch.tensor(scaler.fit_transform(X[:,i].reshape(-1,1))).squeeze()
        scaler_list.append(scaler)
    # print('X after scaler',X)
    
    # print('y before scaler',y)
    scaler2 = StandardScaler()
    y = scaler2.fit_transform(y)
    # valid_data = scaler.transform(valid_data)
    # print('y after scaler',y)
    X = X.double().to(device)
    y = torch.tensor(y).double().to(device)

    # print(X.shape)
    # print(y.shape)
    # assert 0

    # print(X.shape)
    # print(y.shape)
    # fit the model
    if name in ['rf','mlp']:
        X = X.detach().cpu()
        y = y.detach().cpu()
    # print(name)
    
    if name in ['oodgp','gp']:
        model.fit(X, y)
    else:
        model.fit(X, y.ravel())
    # print(X,y)
    # plot before hand

    print('start fitting origin nodes')
    # for epoch in trange(200):
    
    
    if name in ['oodgp','gp']:
        model.train_step(X, y, opt2)


    # plot(X, y,env_list,model,hyper_name+' before_pid.jpg', scaler1 = scaler1, scaler2 = scaler2)

    # assert 0
    # perform the optimization process
    print('start finding hyp')
    
    
    # print('init X: ',X)
    # assert 0
    newenv_result_list = []
    all_result_list = []
    x_list = []
    k = 5
    for i in trange(opt.epoch):
        env = generate_env()
        env_list.append(env)
        # print( "length: ", oodgp.length_scale.detach().cpu())
        # print( "noise: ", oodgp.noise_scale.detach().cpu())
        # print( "amplitude: ", oodgp.amplitude_scale.detach().cpu())

        # select the next point to sample
        x = opt_acquisition(X, y, model,scale_list,scaler_list = scaler_list,model_name = name).reshape(1,-1)
        # print(x.shape)
        # print('choose1: ',x)
        # x =  torch.tensor([[x]]).to(device)
        # sample the point
        ori_x = inverse_scale_x(x.detach().cpu(),scaler_list)
        # print('choose: ',x)
        # print(ori_x.shape)
        actual = objective(ori_x.squeeze(),env,record=0,name=name,test_name=test_name)
        # print('choose2: ',x)
        # newenv_result = objective(ori_x.squeeze(),env = 1,record=0,name=name,test_name=test_name)
        # newenv_result_list.append(newenv_result)
        
        # summarize the finding
        est , _ = surrogate(model,x,model_name = name)
        # print('est shape: ',est.shape)
        # print(x)
        # print(scaler2.inverse_transform(est.detach().cpu()))
        # print(actual)

        if (name in ['mlp','rf']):
            pass
        else:
            # print(est)
            est =est.detach().cpu()
        # print('choose3: ',x)
        print(f'>x={x.detach().cpu()}, ori_x={ori_x}, f()={scaler2.inverse_transform(est.reshape(-1,1))}, actual={actual}')
        # print('>x=%.3f, f()=%3f, actual=%.3f' % (x, scaler2.inverse_transform(est.detach().cpu()), actual))
        # add the data to the dataset
        # print('choose: ',x)
        X = torch.cat((X, x),dim = 0)
        # print('after cat: ',X)
        # assert 0
        # print(y.shape)
        # print(torch.tensor(scaler2.transform(np.array([actual]).reshape(1,-1))).to(device).shape)
       
        
        y = torch.cat((y, torch.tensor(scaler2.transform(np.array([actual]).reshape(1,-1))).to(device)),dim = 0)
        
        ix = argmin(y.cpu())
        X_ori = inverse_scale_x(X.detach().cpu(),scaler_list)
        y_ori = scaler2.inverse_transform(y.cpu())
        
        best_x = X_ori[ix]
        best_y = y_ori[ix]
        new_env_best_y = objective(best_x.squeeze(),env = 1,record=0,name=name,test_name=test_name)
        
        x_list.append(best_x.detach().cpu().numpy())
        all_result_list.append(best_y)
        newenv_result_list.append(new_env_best_y)
        
        
        # update the model
        if (name in ['oodgp','gp']) and ((i-1) % k == 0):
            model.train_step(X, y, opt2)


    if(not os.path.exists(('new_logs/'+opt.save_name+'/'+test_name+'/'))):
        os.makedirs('new_logs/'+opt.save_name+'/'+test_name+'/')
                    
    np.save('new_logs/'+opt.save_name+'/'+test_name+'/'+name+'_iteration_newenv.npy',newenv_result_list)
    np.save('new_logs/'+opt.save_name+'/'+test_name+'/'+name+'_iteration_allenv.npy',all_result_list)
    np.save('new_logs/'+opt.save_name+'/'+test_name+'/'+name+'_iteration_x.npy',x_list)
  
    # print('check env: ',env_list)
    # plot all samples and the final surrogate function
    # plot(X, y, env_list,model,hyper_name+' after.jpg',scaler1 = scaler1, scaler2 = scaler2)
    # best result
    ix = argmin(y.cpu())
    # print('final X: ',X)
    # print(X.shape)
    X_ori = inverse_scale_x(X.detach().cpu(),scaler_list)
    # y_ori = scaler2.inverse_transform(y.cpu())
    
    # print('Xori: ',X_ori)
    best_x = X_ori[ix]

    best_y = objective(best_x,env=1,record=1,name=name,test_name=test_name)
    
    print(f'Best Result: x= {best_x}, y= {best_y}')
    
    return best_x, best_y



if __name__ == '__main__':

    test_name_list = ['sin_forward','spiral_up','hover', 'fig-8']

  
    result_record = {}
    
    test_name = opt.t_name
    
    # for test_name in test_name_list:
    
    if(opt.model == 'oodgp'):
        
        if(opt.kernel == 'nokernel'):
            oodgp = OODGP(opt).to(device)
            best_x, best_y = BayesianOptimization(model = oodgp,name = 'oodgp',test_name=test_name)
            result_record['oodgp+'+test_name] = (best_x, best_y)
            print(f'oodgp+{test_name}: best_x:{best_x},best_y:{best_y}')
        
        elif(opt.kernel == 'RationalQuadraticKernel'):
            kernel = RationalQuadraticKernel().to(opt.device)
            oodgp = OODGPKernel(kernel, opt.envlr, 1, opt.lambdae, True).to(opt.device)
            best_x, best_y = BayesianOptimization(model = oodgp,name = 'oodgp',test_name=test_name)
            result_record['oodgp_rq+'+test_name] = (best_x, best_y)
            print(f'oodgp_rq+{test_name}: best_x:{best_x},best_y:{best_y}')
          
        elif(opt.kernel == 'MaternKernel25'):
            kernel = MaternKernel25().to(opt.device)
            oodgp = OODGPKernel(kernel, opt.envlr, 1, opt.lambdae, True).to(opt.device)
            best_x, best_y = BayesianOptimization(model = oodgp,name = 'oodgp',test_name=test_name)
            result_record['oodgp_mk+'+test_name] = (best_x, best_y)
            print(f'oodgp_mk+{test_name}: best_x:{best_x},best_y:{best_y}')
              
        elif(opt.kernel == 'DotProductKernel'):
            kernel = DotProductKernel().to(opt.device)
            oodgp = OODGPKernel(kernel, opt.envlr, 1, opt.lambdae, True).to(opt.device)
            best_x, best_y = BayesianOptimization(model = oodgp,name = 'oodgp',test_name=test_name)
            result_record['oodgp_rq+'+test_name] = (best_x, best_y)
            print(f'oodgp_rq+{test_name}: best_x:{best_x},best_y:{best_y}')
        
    elif(opt.model == 'gp'):
        gp = GP().to(device)
        best_x, best_y = BayesianOptimization(model = gp,name = 'gp',test_name=test_name)
        print(f'gp+{test_name}: best_x:{best_x},best_y:{best_y}')
        result_record['gp+'+test_name] = (best_x, best_y)
    
    elif(opt.model == 'mlp'):
        mlpregr = MLPRegressor(random_state=1, max_iter=500)
        best_x, best_y = BayesianOptimization(model = mlpregr,name = 'mlp',test_name=test_name)
        print(f'mlp+{test_name}: best_x:{best_x},best_y:{best_y}')
        result_record['mlp+'+test_name] = (best_x, best_y)       
    
    elif(opt.model == 'rf'):
        rfregr = RandomForestRegressor(n_estimators=50, random_state=0)
        best_x, best_y = BayesianOptimization(model = rfregr,name = 'rf',test_name=test_name)
        print(f'rf+{test_name}: best_x:{best_x},best_y:{best_y}')
        result_record['rf+'+test_name] = (best_x, best_y)
        
    else:
        print(opt.model)
        assert 0
        
    print(result_record)