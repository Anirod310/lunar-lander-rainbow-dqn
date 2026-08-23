import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class NoisyLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NoisyLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.mu_w = nn.Parameter(torch.empty(output_dim, input_dim))
        self.sigma_w = nn.Parameter(torch.empty(output_dim, input_dim))

        self.mu_b = nn.Parameter(torch.empty(output_dim))
        self.sigma_b = nn.Parameter(torch.empty(output_dim))

        self.register_buffer("w_epsilon", torch.empty(output_dim, input_dim))
        self.register_buffer("b_epsilon", torch.empty(output_dim))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.input_dim)
        self.mu_w.data.uniform_(-bound, +bound)
        self.mu_b.data.uniform_(-bound, +bound)

        self.sigma_w.data.fill_(0.5 / math.sqrt(self.input_dim))
        self.sigma_b.data.fill_(0.5 / math.sqrt(self.input_dim))

    def reset_noise(self):
        input_noise = torch.randn(self.input_dim)
        input_epsilon = input_noise.sign() * input_noise.abs().sqrt()

        output_noise = torch.randn(self.output_dim)
        output_epsilon = output_noise.sign() * output_noise.abs().sqrt()

        self.b_epsilon.copy_(output_epsilon)
        self.w_epsilon.copy_(torch.outer(output_epsilon, input_epsilon))

    def forward(self, x):
        weight = self.mu_w + (self.sigma_w * self.w_epsilon)
        bias = self.mu_b + (self.sigma_b * self.b_epsilon)

        return F.linear(x, weight, bias)


class Rainbow(nn.Module):
    def __init__(self, input_dim, output_dim, n_atoms):
        super(Rainbow, self).__init__()

        self.action_dim = output_dim
        self.n_atoms = n_atoms

        self.register_buffer("Z_tensor", torch.linspace(-200, 200, n_atoms))

        self.feature_layer = nn.Sequential(nn.Linear(input_dim, 128),
                                      nn.ReLU())
        self.value_stream = nn.Sequential(NoisyLinear(128, 128),
                                     nn.ReLU(),
                                     NoisyLinear(128, n_atoms))
        self.advantage_stream = nn.Sequential(NoisyLinear(128, 128),
                                         nn.ReLU(),
                                         NoisyLinear(128, output_dim * n_atoms))

    def forward(self, x):
        features = self.feature_layer(x)
        v= self.value_stream(features)
        a = self.advantage_stream(features)

        batch_size = x.size(0)
        v = v.view(batch_size, 1, self.n_atoms)
        a = a.view(batch_size, self.action_dim, self.n_atoms)

        q_values = v + (a - torch.mean(a, dim=1, keepdim=True))
        probs = F.softmax(q_values, dim=-1)

        return probs

    def reset_all_noise(self):
        self.value_stream[0].reset_noise()
        self.value_stream[2].reset_noise()
        self.advantage_stream[0].reset_noise()
        self.advantage_stream[2].reset_noise()





        
        
