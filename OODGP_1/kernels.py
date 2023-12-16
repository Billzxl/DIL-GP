import numpy as np
import torch
import torch.nn as nn

class RationalQuadraticKernel(nn.Module):
    def __init__(self, length_scale=1.0, alpha=1.0, amplitude_scale=1.0):
        super().__init__()
        # self.length_scale = length_scale
        # self.alpha = alpha
        self.amplitude_scale_ = nn.Parameter(torch.tensor(np.log(amplitude_scale)))
        self.length_scale_ = nn.Parameter(torch.tensor(np.log(length_scale)))
        self.alpha_ = nn.Parameter(torch.tensor(np.log(alpha)))
        # self.alpha_ = 1
    
    @property
    def amplitude_scale(self):
        return torch.exp(self.amplitude_scale_)
    
    @property
    def length_scale(self):
        return torch.exp(self.length_scale_)    

    @property
    def alpha(self):
        return torch.exp(self.alpha_)
        # return np.exp(self.alpha_)
    
    def forward(self, X, Z, detach=False):
        Xsq = (X**2).sum(dim=1, keepdim=True)
        Zsq = (Z**2).sum(dim=1, keepdim=True)
        sqdist = Xsq + Zsq.T - 2 * X.mm(Z.T)
        if not detach:
            return self.amplitude_scale * (1 + sqdist / (2 * self.alpha * self.length_scale)).pow(-self.alpha)
        else:
            return self.amplitude_scale.detach() * (1 + sqdist / (2 * self.alpha * self.length_scale.detach())).pow(-self.alpha.detach())
    
    def forward_w_scale(self, X, Z, scale, detach=False):
        if detach:
            amplitude_scale = self.amplitude_scale.detach() * scale
            length_scale = self.length_scale.detach() * scale
            # alpha = self.alpha.detach() * scale
        else:
            amplitude_scale = self.amplitude_scale * scale
            length_scale = self.length_scale * scale    
            # alpha = self.alpha * scale
        
        alpha = self.alpha

        Xsq = (X**2).sum(dim=1, keepdim=True)
        Zsq = (Z**2).sum(dim=1, keepdim=True)
        sqdist = Xsq + Zsq.T - 2 * X.mm(Z.T)
        return amplitude_scale * (1 + sqdist / (2 * alpha * length_scale)).pow(-alpha)

class ExponentiationKernel(nn.Module):
    def __init__(self, p_scale=1.0, amplitude_scale=1.0):
        super().__init__()
        # self.length_scale = length_scale
        # self.alpha = alpha
        self.amplitude_scale_ = nn.Parameter(torch.tensor(np.log(amplitude_scale)))
        self.p_scale_ = nn.Parameter(torch.tensor(np.log(p_scale)))
        
    @property
    def amplitude_scale(self):
        return torch.exp(self.amplitude_scale_)

    @property
    def p_scale(self):
        return torch.exp(self.p_scale_)
    
    def forward(self, X, Z, detach=False):
        Xsq = (X**2).sum(dim=1, keepdim=True)
        Zsq = (Z**2).sum(dim=1, keepdim=True)
        sqdist = Xsq + Zsq.T - 2 * X.mm(Z.T)
        if not detach:
            return
        else:
            return
    

class DotProductKernel(nn.Module):
    def __init__(self, sigma_zero=1.0, amplitude_scale=1.0):
        super().__init__()
        # self.length_scale = length_scale
        # self.alpha = alpha
        self.amplitude_scale_ = nn.Parameter(torch.tensor(np.log(amplitude_scale)))
        self.sigma_zero_ = nn.Parameter(torch.tensor(np.log(sigma_zero)))
        
    @property
    def amplitude_scale(self):
        return torch.exp(self.amplitude_scale_)

    @property
    def sigma_zero(self):
        return torch.exp(self.sigma_zero_)
    
    def forward(self, X, Z, detach=False):
        K = X.mm(Z.T)
        if not detach:
            return self.amplitude_scale * (K + self.sigma_zero)
        else:
            return self.amplitude_scale.detach() * (K + self.sigma_zero.detach())
    
    def forward_w_scale(self, X, Z, scale, detach=False):
        if detach:
            amplitude_scale = self.amplitude_scale.detach() * scale
            sigma_zero = self.sigma_zero.detach() * scale
        else:
            amplitude_scale = self.amplitude_scale * scale
            sigma_zero = self.sigma_zero * scale
        
        K = X.mm(Z.T)
        return amplitude_scale * (K + sigma_zero)