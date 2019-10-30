from DynamicalSystemLayer import*
from VaeNeuralODELayer import*

class EncoderFC(nn.Module):
    def __init__(self, input_dim, hidden_dim, param_dim):
        super(EncoderFC, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.param_dim = param_dim

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.selu = nn.SELU()
        self.lin2 = nn.Linear(self.hidden_dim, self.param_dim)

    def forward(self, x):
        x = x.view(-1)
        out = self.lin1(x)
        out = self.selu(out)
        out = self.lin2(out)
        out = self.selu(out)
        out = out.view(1,len(out))
        return out

class DecoderVaeODE(nn.Module):
    def __init__(self, hidden_dim, input_dim):
        super(DecoderVaeODE, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.selu = nn.SELU()
        self.lin2 = nn.Linear(self.hidden_dim, self.input_dim)

    def forward(self, z0, t, p):
        func = LinearODEF(p)
        self.ode = VaeNeuralODE(func)
        z = self.ode(z0, t, return_whole_sequence=True)
        y = z.squeeze(1)
        y = y[:,0]
        out = self.lin1(y)
        out = self.selu(out)
        out = self.lin2(out)
        out = self.selu(out)
        out = out.view(len(out), 1)
        return out

class VaeODENeuralNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, param_dim):
        super(VaeODENeuralNet, self).__init__()
        self.input_dim = input_dim
        self.param_dim = param_dim
        self.hidden_dim = hidden_dim

        self.param_encoder = EncoderFC(self.input_dim, self.hidden_dim, self.param_dim)
        self.ode_decoder = DecoderVaeODE(self.hidden_dim, self.input_dim)

    def forward(self, x, t):
        x = x.view(1,len(x))
        p = self.param_encoder(x)
        z0 = torch.Tensor([[x[:,0], x[:,1]]])
        z = self.ode_decoder(z0, t, p)
        return z, p

class SimpleVaeODENeuralNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, param_dim):
        super(SimpleVaeODENeuralNet, self).__init__()
        self.input_dim = input_dim
        self.param_dim = param_dim
        self.hidden_dim = hidden_dim

        self.param_encoder = EncoderFC(self.input_dim, self.hidden_dim, self.param_dim)

    def forward(self, x, t):
        x = x.view(1,len(x))
        p = self.param_encoder(x)
        z0 = torch.Tensor([[x[:,0], x[:,1]]])
        func = LinearODEF(p)
        self.ode = VaeNeuralODE(func)
        z = self.ode(z0, t, return_whole_sequence=True)
        y = z.squeeze(1)
        out = y[:,0]
        out = out.view(len(out), 1)
        return out, p

class VaePendulumNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, param_dim, func):
        super(VaePendulumNet, self).__init__()
        self.input_dim = input_dim
        self.param_dim = param_dim
        self.hidden_dim = hidden_dim

        self.func = func

        self.param_encoder = EncoderFC(self.input_dim, self.hidden_dim, self.param_dim)

    def forward(self, x, t):
        x = x.view(1, len(x))
        k = self.param_encoder(x)
        p = torch.Tensor([[0., 1., k, 0.]])
        fun = self.func(p)
        self.ode = VaeNeuralODE(fun)
        z = self.ode(z0, t, return_whole_sequence=True)
        y = z.squeeze(1)
        out = y[:,0]
        out = out.view(len(out), 1)
        return out, p


