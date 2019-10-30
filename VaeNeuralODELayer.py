from DynamicalSystemLayer import *

class LinearODEF(VaeODEF):
    def __init__(self, p):
        super(LinearODEF, self).__init__()
        self.p = p.detach().requires_grad_(True)
        self.mat_dim = int(self.p.shape[1]**0.5)

    def forward(self, x, t):
        A = self.p.reshape(self.mat_dim, self.mat_dim)
        out = x.matmul(A)
        return out
