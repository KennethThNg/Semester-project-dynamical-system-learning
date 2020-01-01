import torch.nn as nn
from DynamicalSystemLayer import *

class LinearODEModel(nn.Module):
    def __init__(self, in_feature, out_feature, bias=False):
        '''
        Build a neural net containing the linear dynamical system layer
        :param in_feature (int): input dimension
        :param out_feature (int): output dimension
        :param bias (bool): determine if bias is needed in the model
        '''
        super(LinearODEModel, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.bias = bias
        self.ode = LinearODELayer(self.in_feature, self.out_feature, self.bias)

    def forward(self, x):
        '''
        Forward pass
        :param x (torch.Tensor): input data. Tensor of dimension [batch size, in_feature]
        :return:
        out (torch.Tensor): output feature. Tensor of dimension [batch size, out_feature]
        '''
        out = self.ode(x)
        return out



class NNODEModel(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, bias=False):
        '''
        Build a neural net containing the linear dynamical system layer and the non-linear dynamical system layer
        :param in_dim (int): Input dimension
        :param hid_dim (int): Hidden dimension
        :param out_dim (int): Output dimension
        :param bias (bool): determine if bias is needed in the model
        '''
        super(NNODEModel, self).__init__()
        #Dimension
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim

        self.bias = bias

        #Layer
        self.lin_ode = LinearODELayer(self.in_dim, self.out_dim, self.bias)
        self.nl_ode = NonLinearODELayer(self.in_dim, self.hid_dim, self.out_dim)

        #weight init
        self.lin_ode.weight.data.uniform_(-0.1,0.1)
        if self.bias:
            self.lin_ode.bias.data.uniform_(-0.1,0.1)
        self.nl_ode.weight.data.uniform_(-0.001,0.001)


    def forward(self, x):
        '''
        Forward Pass
        :param x (torch.Tensor): input data. Tensor of dimension [batch size, in_feature]
        :return:
        out (torch.Tensor): output feature. Tensor of dimension [batch size, out_feature]
        '''
        out = self.lin_ode(x) + self.nl_ode(x)
        return out

