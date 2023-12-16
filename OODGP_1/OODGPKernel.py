import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch import autograd
from sklearn.cluster import KMeans
from tqdm import tqdm


class OODGPKernel(nn.Module):
    def __init__(self, kernel, envlr=1e-3, eistep=3, lambdae=1e-10, use_kmeans=True):
        super().__init__()
        self.kernel = kernel
        self.scale = torch.tensor(1.).requires_grad_()
        self.noise_scale_ = nn.Parameter(torch.tensor(np.log(1.0)))

        self.envlr = envlr
        self.eistep = eistep
        self.lambdae = lambdae
        self.use_kmeans = use_kmeans
        self.env_labels = None

        self.env_w = None

    @property
    def noise_scale(self):
        return torch.exp(self.noise_scale_)
    
    def forward(self, x):
        """compute prediction. fit() must have been called.
        x: test input data point. N x D tensor for the data dimensionality D."""
        mx = 0
        y = self.y
        L = self.L
        alpha = self.alpha
        k = self.kernel(self.X, x)
        v = torch.linalg.solve(L, k)
        mu = mx + k.T.mm(alpha)
        var = self.kernel.amplitude_scale + self.noise_scale - torch.diag(v.T.mm(v))
        return mu, var

    def fit(self, X, y):
        """should be called before forward() call.
        X: training input data point. N x D tensor for the data dimensionality D.
        y: training target data point. N x 1 tensor."""
        
        #define learning rate, env labels, and optimizer 

        if self.env_w is None:
            if not self.use_kmeans:
                self.env_w = torch.randn(len(X), device=X.device, requires_grad = True)
            else:
                kmeans = KMeans(n_clusters=2, random_state=0).fit(X.cpu().numpy())
                env_labels = kmeans.labels_ * 2 - 1
                self.env_labels = env_labels
                self.env_w = torch.tensor(env_labels, dtype=torch.float32, device=X.device).requires_grad_()
        else:
            self.env_w = torch.randn(len(X), device=X.device, requires_grad = True)

        env_w = self.env_w
        opti_env = optim.Adam([env_w], lr=self.envlr)

        D = X.shape[1]
        if len(y.shape) == 1:
            y = y.unsqueeze(1)

        mx = torch.zeros_like(y)
        
        #========Update on the environment splitting according to the IRM algorithm
        ei_step = self.eistep
        cl_step = 1
        for i in range(ei_step):
            self.scale = torch.tensor(1., device=X.device).requires_grad_()
            K_detach = self.kernel.forward_w_scale(X, X, self.scale, detach=True) + (self.noise_scale.detach() * self.scale) * torch.eye(len(X), device=X.device)
            try:
                L_dtc = torch.linalg.cholesky(K_detach)
            except:
                L_dtc = torch.linalg.cholesky(K_detach + 1e-3 * torch.eye(len(K_detach), device=X.device))
            
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
        K = self.kernel.forward_w_scale(X, X, self.scale) + (self.noise_scale * self.scale) * torch.eye(len(X), device=X.device)
        try:
            L = torch.linalg.cholesky(K)
        except:
            L = torch.linalg.cholesky(K + 1e-3 * torch.eye(len(X), device=X.device))

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
            marginal_likelihood = -0.5 * (y-mx).T.mm(alpha) - torch.log(torch.diag(L)).sum() - 0.5 * D * np.log(2 * np.pi)

            # print('marginal_likelihood + npenalty', marginal_likelihood , npenalty)
            marginal_likelihood = marginal_likelihood + npenalty * self.lambdae
                                   
        self.X = X
        self.y = y
        self.L = L
        self.alpha = alpha
        self.K = K
        
        self.env_w = env_w
        
        return marginal_likelihood

    def train_step(self, X, y, opt, e=None):
        """gradient-based optimization of hyperparameters
        opt: torch.optim.Optimizer object."""
        opt.zero_grad()
        
        nll = -self.fit(X, y).sum() / len(X)
        nll.backward()

        opt.step()
        return {
            "loss": nll.item(),
            # "length": self.length_scale.detach().cpu(),
            "noise": self.noise_scale.detach().cpu(),
            # "amplitude": self.amplitude_scale.detach().cpu(),
        }
