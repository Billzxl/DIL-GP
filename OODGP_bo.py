import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch import autograd
from sklearn.cluster import KMeans
# from oodgp_BayesianOptimization import opt

import argparse

lambdae = 0.1
# parser1 = argparse.ArgumentParser(description='OODGP BO')


# parser1.add_argument('--lambdae',
#                     help='lr outside',
#                     default=0.1,#[0.1,1e-5]
#                     type=float)



# opt1 = parser1.parse_args()


# device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


class OODGP(nn.Module):
    def __init__(self,opt, envlr=1e-3, eistep=1, lambdae=lambdae):
        super().__init__()
        
        length_scale=1.0; noise_scale=1.0; amplitude_scale=1.0
        
        # self.length_scale_ = nn.Parameter(torch.tensor(length_scale))
        # self.noise_scale_ = nn.Parameter(torch.tensor(noise_scale))
        # self.amplitude_scale_ = nn.Parameter(torch.tensor(amplitude_scale))
        
        self.length_scale_ = nn.Parameter(torch.tensor(np.log(length_scale)))
        self.noise_scale_ = nn.Parameter(torch.tensor(np.log(noise_scale)))
        self.amplitude_scale_ = nn.Parameter(torch.tensor(np.log(amplitude_scale)))
        self.scale = torch.tensor(1.).requires_grad_()

        self.device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")
        self.envlr = opt.envlr
        self.eistep = eistep
        self.lambdae = opt.lambdae
        

        
    # @property
    # def length_scale(self):
    #     return (self.length_scale_**2)

    # @property
    # def noise_scale(self):
    #     return (self.noise_scale_**2)

    # @property
    # def amplitude_scale(self):
    #     return (self.amplitude_scale_**2)    
        
    @property
    def length_scale(self):
        return torch.exp(self.length_scale_)

    @property
    def noise_scale(self):
        return torch.exp(self.noise_scale_)

    @property
    def amplitude_scale(self):
        return torch.exp(self.amplitude_scale_)

    def forward(self, x):
        """compute prediction. fit() must have been called.
        x: test input data point. N x D tensor for the data dimensionality D."""
        mx = 0
        y = self.y
        L = self.L
        alpha = self.alpha
        k = self.kernel_mat(self.X, x)
        v = torch.linalg.solve(L, k)
        mu = mx + k.T.mm(alpha)
        var = self.amplitude_scale + self.noise_scale - torch.diag(v.T.mm(v))
        return mu, var

    def fit2(self, X, y):
        """should be called before forward() call.
        Inp: training input data point. N x D tensor for the data dimensionality D.
        y: training target data point. N x 1 tensor."""

        #define learning rate, env labels, and optimizer 
        # lr=0.001
        env_w = torch.randn(len(X)).requires_grad_()
        # opti_env = optim.Adam([env_w], lr=lr) # why optimization
        opti_env = optim.SGD([env_w], lr=0.01)
        
        D = X.shape[1]
        K = self.kernel_mat_self(X)
        mx = 0
        L = torch.linalg.cholesky(K) # may be not positive-determinant ?
        
        #========Update on the environment splitting according to the IRM algorithm
        for _ in range(2):
            alpha = torch.linalg.solve(L.T, torch.linalg.solve(L, (y-mx)*env_w.sigmoid()))
            likelia = -0.5 * ( (y-mx)*env_w.sigmoid() ).T.mm(alpha) - torch.log(torch.diag(L)).sum() - D * 0.5 * np.log(2 * np.pi)
            likelia = likelia.mean()
            grada = autograd.grad(likelia, [self.scale], create_graph=True)[0]
            penaltya = torch.sum(grada**2)
            
            alpha = torch.linalg.solve(L.T, torch.linalg.solve(L, (y-mx)*(1-env_w.sigmoid())))
            likelib = -0.5 * ( (y-mx)*(1-env_w.sigmoid()) ).T.mm(alpha) - torch.log(torch.diag(L)).sum() - D * 0.5 * np.log(2 * np.pi)
            likelib  = likelib.mean()
            gradb = autograd.grad(likelib, [self.scale], create_graph=True)[0]
            penaltyb = torch.sum(gradb**2)
            
            npenalty = - torch.stack([penaltya, penaltyb]).mean()  
            # npenalty = 0.0                       
            opti_env.zero_grad()
            npenalty.backward()
            opti_env.step()
        
        #========Maximum likelihood calculation
        #1)Calculate the IRM penalty based on the updated env variable
        alpha = torch.linalg.solve(L.T, torch.linalg.solve(L, (y-mx)*env_w.sigmoid()))
        lossa = -0.5 * ( (y-mx)*env_w.sigmoid() ).T.mm(alpha) - torch.log(torch.diag(L)).sum() - D * 0.5 * np.log(2 * np.pi)
        lossa = -lossa.mean()
        grada = autograd.grad(lossa, [self.scale], create_graph=True)[0]
        penaltya = torch.sum(grada**2)
        
        
        alpha = torch.linalg.solve(L.T, torch.linalg.solve(L, (y-mx)*(1-env_w.sigmoid())))
        lossb = -0.5 * ( (y-mx)*(1-env_w.sigmoid()) ).T.mm(alpha) - torch.log(torch.diag(L)).sum() - D * 0.5 * np.log(2 * np.pi)
        lossb = -lossb.mean()
        gradb = autograd.grad(lossb, [self.scale], create_graph=True)[0]
        penaltyb = torch.sum(gradb**2)
        
        npenalty = - torch.stack([penaltya, penaltyb]).mean()       
        
        #2)Sum up the original likelihood and the penalty
        alpha = torch.linalg.solve(L.T, torch.linalg.solve(L, (y-mx)))
        marginal_likelihood = -0.5 * ( (y-mx) ).T.mm(alpha) - torch.log(torch.diag(L)).sum() - D * 0.5 * np.log(2 * np.pi)
        marginal_likelihood = marginal_likelihood + 1e-10 * npenalty
                                   
        self.X = X
        self.y = y
        self.L = L
        self.alpha = alpha
        self.K = K
        
        self.env_w = env_w
        
        return marginal_likelihood

    def fit(self, X, y):
        """should be called before forward() call.
        X: training input data point. N x D tensor for the data dimensionality D.
        y: training target data point. N x 1 tensor."""
        
        #define learning rate, env labels, and optimizer 
        self.scale = torch.tensor(1., device=X.device).requires_grad_()

        # env_w = torch.randn(len(X),requires_grad = True)
        kmeans = KMeans(n_clusters=2, random_state=0,n_init=10).fit(X.cpu().numpy())
        env_labels = kmeans.labels_ * 2 - 1
        env_w = torch.tensor(env_labels, dtype=torch.float32, device=X.device).requires_grad_()
        opti_env = optim.Adam([env_w], lr=self.envlr)

        D = X.shape[1]
        if len(y.shape) == 1:
            y = y.unsqueeze(1)

        mx = torch.zeros_like(y)
        
        #========Update on the environment splitting according to the IRM algorithm
        ei_step = self.eistep
        cl_step = 1
        for i in range(ei_step):
            K_detach = self.kernel_mat_self(X, detach=True)
            # print(K_detach)
            L_dtc = torch.linalg.cholesky(K_detach)
            alpha = torch.linalg.solve(L_dtc.T, torch.linalg.solve(L_dtc, (y-mx)*env_w.sigmoid()))
            likelia = -0.5 * ( (y-mx)*env_w.sigmoid() ).T.mm(alpha) - \
                        torch.log(torch.diag(L_dtc)).sum() - D * 0.5 * np.log(2 * np.pi)
            likelia = likelia.mean()
            grada = autograd.grad(likelia, [self.scale], create_graph=True)[0]
            penaltya = torch.sum(grada**2)
            
            
            alpha = torch.linalg.solve(L_dtc.T, torch.linalg.solve(L_dtc, (y-mx)*(1-env_w.sigmoid())))
            likelib = -0.5 * ( (y-mx)*(1-env_w.sigmoid()) ).T.mm(alpha) - \
                        torch.log(torch.diag(L_dtc)).sum() - D * 0.5 * np.log(2 * np.pi)
            likelib  = likelib.mean()
            gradb = autograd.grad(likelib, [self.scale], create_graph=True)[0]
            penaltyb = torch.sum(gradb**2)
            
            npenalty = - torch.stack([penaltya, penaltyb]).mean()                         
            opti_env.zero_grad()
            npenalty.backward()
            opti_env.step()

        #========Maximum likelihood calculation
        self.scale = torch.tensor(1., device=X.device).requires_grad_()
        K = self.kernel_mat_self(X)
        L = torch.linalg.cholesky(K)
        env_w = env_w.detach() #for test
        #1)Calculate the IRM penalty based on the updated env variable
        for i in range(cl_step):
            alpha = torch.linalg.solve(L.T, torch.linalg.solve(L, (y-mx)*env_w.sigmoid()))
            lossa = -0.5 * ( (y-mx)*env_w.sigmoid() ).T.mm(alpha) - torch.log(torch.diag(L)).sum() - D * 0.5 * np.log(2 * np.pi)
            lossa = -lossa.mean()
            grada = autograd.grad(lossa, [self.scale], create_graph=True)[0]
            penaltya = torch.sum(grada**2)
            
            
            alpha = torch.linalg.solve(L.T, torch.linalg.solve(L, (y-mx)*(1-env_w.sigmoid())))
            lossb = -0.5 * ( (y-mx)*(1-env_w.sigmoid()) ).T.mm(alpha) - torch.log(torch.diag(L)).sum() - D * 0.5 * np.log(2 * np.pi)
            lossb = -lossb.mean()
            gradb = autograd.grad(lossb, [self.scale], create_graph=True)[0]
            penaltyb = torch.sum(gradb**2)
            
            npenalty = - torch.stack([penaltya, penaltyb]).mean()       
            
            #2)Sum up the original likelihood and the penalty
            alpha = torch.linalg.solve(L.T, torch.linalg.solve(L, (y-mx)))
            marginal_likelihood = -0.5 * ( (y-mx) ).T.mm(alpha) - torch.log(torch.diag(L)).sum() - D * 0.5 * np.log(2 * np.pi)
            marginal_likelihood = marginal_likelihood + npenalty * self.lambdae

            # print('marginal_likelihood, npenalty', marginal_likelihood, npenalty)
                                   
        self.X = X
        self.y = y
        self.L = L
        self.alpha = alpha
        self.K = K
        
        self.env_w = env_w
        
        return marginal_likelihood

    def kernel_mat_self(self, X, detach=False):
        sq = (X**2).sum(dim=1, keepdim=True)
        sqdist = sq + sq.T - 2 * X.mm(X.T)

        if not detach:
            return (self.amplitude_scale *self.scale) * torch.exp(
                -0.5 * sqdist / (self.length_scale*self.scale)
            ) + (self.noise_scale * self.scale) * torch.eye(len(X)).to(self.device)
        else:
            return (self.amplitude_scale.detach() *self.scale) * torch.exp(
                -0.5 * sqdist / (self.length_scale.detach()*self.scale)
            ) + (self.noise_scale.detach() * self.scale) * torch.eye(len(X)).to(self.device)

    def kernel_mat(self, X, Z):
        # print(X.shape)
        # print(type(X))
        # print(Z.shape)
        # print(type(Z))
        Xsq = (X**2).sum(dim=1, keepdim=True)
        Zsq = (Z**2).sum(dim=1, keepdim=True)
        sqdist = Xsq + Zsq.T - 2 * X.mm(Z.T)
        return self.amplitude_scale *self.scale * torch.exp(-0.5 * sqdist / (self.length_scale*self.scale))

    def train_step(self, X, y, opt, e=None):
        """gradient-based optimization of hyperparameters
        opt: torch.optim.Optimizer object."""
        opt.zero_grad()
        
        nll = -self.fit(X, y).sum()/len(X)
        nll.backward()

        opt.step()
        return {
            "loss": nll.item(),
            "length": self.length_scale.detach().cpu(),
            "noise": self.noise_scale.detach().cpu(),
            "amplitude": self.amplitude_scale.detach().cpu(),
        }