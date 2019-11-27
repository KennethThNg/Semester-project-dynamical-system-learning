import torch
import torch.nn as nn
from torch.autograd import *
import math

class LinearODELayer(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
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
        output = x.mm(self.weight)
        if self.bias is not None:
            output += self.bias
        return output
    # the backward function is not necessary, sincewe have only used
    # functions from pytorch (in particular, .reshape and .mm)
    
    # this function is inherited from the pytorch class Linear
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            
#Thsi object can be extended
class NonLinearODELayer(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(NonLinearODELayer, self).__init__()
        self.in_dim = in_dim
        self.weight = nn.Parameter(torch.Tensor(self.in_dim, self.in_dim, self.in_dim))
    
    def forward(self, x):
        xBx = x.matmul(self.weight).matmul(x.t()).permute(2,1,0)
        out = torch.diagonal(xBx).t()
        return out
    
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.srqt(5))
        
class LinearODEModel(nn.Module):
    def __init__(self, in_feature, out_feature, bias=False):
        super(LinearODEModel, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.bias = bias
        self.ode = LinearODELayer(self.in_feature, self.out_feature, self.bias)
        
    def forward(self, x):
        out = self.ode(x)
        return out
    
    
class NNODEModel(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, bias=False):
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
        out = self.lin_ode(x) + self.nl_ode(x)
        return out

def to_np(x):
    return x.detach().cpu().numpy()

def ode_solve(z0, t0, t1, f):
    """
    Simplest Euler ODE initial value solver
    """
    h_max = 0.05
    n_steps = math.ceil((abs(t1 - t0)/h_max).max().item())

    h = (t1 - t0)/n_steps
    t = t0
    z = z0

    for i_step in range(n_steps):
        z = z + h * f(z, t)
        t = t + h
    return z

class VaeODEF(nn.Module):
    def forward_grad(self, z, t, grad_out):
        batch_size = z.shape[0]
        out = self.forward(z, t)

        a = grad_out
        adfdz, *adfdp = torch.autograd.grad((out,), (z, self.p) , grad_outputs=(a), allow_unused=True, retain_graph=True)

        if adfdp is not None:
            adfdp = torch.cat([p_grad.flatten() for p_grad in adfdp]).unsqueeze(0)
            adfdp = adfdp.expand(batch_size, -1) / batch_size

        return adfdz, adfdp


class Dynamical_system_layer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z0, t, flat_parameters, func):
        assert isinstance(func, VaeODEF)
        batch_size, *z_dim = z0.size()
        time_len = t.size(0)

        with torch.no_grad():
            z = torch.zeros(time_len, batch_size, *z_dim).to(z0)
            z[0] = z0
            for it in range(time_len - 1):
                z0 = ode_solve(z0, t[it], t[it+1], func)
                z[it+1] = z0

        ctx.func = func
        ctx.save_for_backward(t, z.clone(), flat_parameters)
        return z

    @staticmethod
    def backward(ctx, grad_out):
        func = ctx.func
        t, z, flat_parameters = ctx.saved_tensors
        time_len, batch_size, *z_dim = z.size()
        n_param = flat_parameters.size(1)

        #grad_out = grad_out.view(time_len, batch_size, z_dim)


        with torch.no_grad():

            adzdp = torch.zeros(time_len, batch_size, *z_dim, n_param).to(grad_out)
            u = adzdp[0]
            for it in range(time_len - 1):

                zi = z[it]
                ti = t[it]

                ai = grad_out[it]

                #zi = zi.view(batch_size, z_dim)
                #ai = ai.view(batch_size, z_dim)
                with torch.set_grad_enabled(True):
                    zi = zi.detach().requires_grad_(True)
                    adfdz, adfdp = func.forward_grad(zi, ti, ai)

                    adfdp = adfdp.to(zi) if adfdp is not None else torch.zeros(batch_size, n_param)

                def dual_dynamic(u, t):
                    return adfdz.matmul(u) + adfdp.unsqueeze(0)

                u = ode_solve(u, ti, t[it+1], dual_dynamic)

                adzdp[it+1] = u

        return adzdp, None, None, None

class VaeNeuralODE(nn.Module):
    def __init__(self, func):
        super(VaeNeuralODE, self).__init__()
        assert isinstance(func, VaeODEF)
        self.func = func
        self.param = self.func.p

    def forward(self, z0, t, return_whole_sequence=False):
        t = t.to(z0)
        z = Dynamical_system_layer.apply(z0, t, self.param, self.func)
        if return_whole_sequence:
            return z
        else:
            return z[-1]

