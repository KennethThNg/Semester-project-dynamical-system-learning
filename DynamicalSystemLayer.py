import torch
import torch.nn as nn
from torch.autograd import *
import math

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

class LinearODEF(VaeODEF):
    def __init__(self, p):
        super(LinearODEF, self).__init__()
        self.p = p.detach().requires_grad_(True)
        self.mat_dim = int(self.p.shape[1]**0.5)

    def forward(self, x, t):
        A = self.p.reshape(self.mat_dim, self.mat_dim)
        out = x.matmul(A)
        return out
