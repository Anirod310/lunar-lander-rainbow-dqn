import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NoisyLinear, self).__init__()
        ...

class Rainbow(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Rainbow, self).__init__()
        self.feature_layer = nn.Sequential(nn.Linear(input_dim, 128),
                                      nn.ReLU())
        self.value_stream = nn.Sequential(nn.Linear(128, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 1))
        self.advantage_stream = nn.Sequential(nn.Linear(128, 128),
                                         nn.ReLU(),
                                         nn.Linear(128, output_dim))

    def forward(self, x):
        features = self.feature_layer(x)
        v= self.value_stream(features)
        a = self.advantage_stream(features)
        q_values = v + (a - torch.mean(a, dim=-1, keepdim=True))

        return q_values


        
        
