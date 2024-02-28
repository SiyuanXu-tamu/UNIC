import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


# Code modified originally from https://vsitzmann.github.io/siren/
class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        #self.act = nn.ReLU()
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
        #return self.act(self.linear(input))


class Siren(nn.Module):
    def __init__(self, features, outermost_linear=False, first_omega_0=30, 
                 hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(features[0], features[1], 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(1, len(features)-2):
            self.net.append(SineLayer(features[i], features[i+1], 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(features[-2], features[-1])
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / features[-2]) / hidden_omega_0, 
                                              np.sqrt(6 / features[-2]) / hidden_omega_0)
                
            self.net.append(final_linear)
            #adding Tanh to original implementation to rescale in [-1, 1]
            self.net.append(nn.Tanh())
        else:
            self.net.append(SineLayer(features[-2], features[-1], 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        output = self.net(coords)
        return output        
