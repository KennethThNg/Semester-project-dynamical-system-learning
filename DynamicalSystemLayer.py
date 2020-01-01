import torch
import torch.nn as nn
from torch.autograd import *
import math

#LAYERS
class LinearODELayer(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        '''
        Linear dynamical system layer
        :param in_features (int): input dimension
        :param out_features (int): output dimension
        :param bias (bool): determine if bias is added in the model
        '''
        super(LinearODELayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    # the forward step
    def forward(self, x):
        '''
        Compute the prediction of the model
        :param x (torch.Tensor): input data. Tensor of dimension [batch_size, in_feature]
        :return:
        output (torch.Tensor): output data. Tensor of dimension [batch_size, out_feature]
        '''
        output = x.mm(self.weight)
        if self.bias is not None:
            output += self.bias
        return output
    # the backward function is not necessary, sincewe have only used
    # functions from pytorch (in particular, .reshape and .mm)
    
    # this function is inherited from the pytorch class Linear
    def reset_parameters(self):
        '''
        reset the weights of the model
        :return: initialized weight
        '''
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            

class NonLinearODELayer(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        '''
        Non Linear dynamical system layer
        :param in_dim (Int): Input dimension
        :param hid_dim (Int): hidden dimension
        :param out_dim (Int): output dimension
        '''
        super(NonLinearODELayer, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.weight = nn.Parameter(torch.Tensor(self.in_dim, self.hid_dim, self.out_dim))
    
    def forward(self, x):
        '''
        Compute the model prediction (forward pass)
        :param x (torch.Tensor): input data. Tensor of dimension [batch size, in_dim]
        :return:
        out (torch.Tensor): predicted values. Tensor of dimension [batch size, out_dim]
        '''
        xBx = x.matmul(self.weight).matmul(x.t()).permute(2,1,0)
        out = torch.diagonal(xBx).t()
        return out
    
    def reset_parameters(self):
        '''

        :return: initialized weight
        '''
        nn.init.kaiming_uniform_(self.weight, a=math.srqt(5))

#-----------------------------------------------------------------------------------
    
