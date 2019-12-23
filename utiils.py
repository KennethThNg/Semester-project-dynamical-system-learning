import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

def generate_pendulum(k, sol, dt=1, T=100, dim=2):
    A = np.asarray([[0,1],[-k,0]])
    I = np.eye(dim)
    AA = (I + dt*A)
    for i in range(T):
        sol.append(np.dot(np.asarray(sol[-1]),AA))

    X = np.asarray(sol)[:-1]
    Y = np.asarray(sol)[1:]
    train_x = torch.FloatTensor(X)
    train_y = torch.FloatTensor(Y)
    return train_x, train_y, torch.Tensor(AA)

def generate_lorenz(sigma, rho, beta, sol, dt=0.01, T=10000, dim=3):
    A = np.asarray([[-sigma, sigma, 0], [rho, -1, 0], [0, 0, -beta]])
    Q0 = np.asarray([[0,0,0], [0,0,0], [0,0,0]])*dt
    Q1= np.asarray([[0,0,-1], [0,0,0], [0,0,0]])*dt
    Q2 = np.asarray([[0,1,0],[0,0,0], [0,0,0]])*dt

    B = np.array([Q0, Q1, Q2])

    I = np.eye(dim)
    AA = (I+dt*A)

    for i in range(T):
        x = np.asarray(sol[-1])
        Ax = AA.dot(x)

        phix = x.T.dot(B).dot(x)

        f = Ax + phix
        sol.append(f)

    X = np.asarray(sol)[:-1]
    Y = np.asarray(sol)[1:]
    train_x = torch.FloatTensor(X)
    train_y = torch.FloatTensor(Y)

    return train_x, train_y, torch.Tensor(AA), torch.Tensor(B)

def tensor_norm(tensor):
    a = tensor.view(-1)
    b = torch.norm(a, p=2)
    return b


def train_model(model, train_x, train_y, loss_fn, optimizer, batch_size):
    train_loss = 0
    for b in range(0, train_x.size(0), batch_size):
        model.zero_grad()
        pred = model(train_x.narrow(0, b, batch_size))
        loss = loss_fn(pred, train_y.narrow(0, b, batch_size))
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    return train_loss, pred

def training_session(model, train_x, train_y, lr=1e-2, n_print=10, n_epoch=100, batch_size=1, print_mat=False, print_feat=False):
    loss_fn = nn.MSELoss()
    optimizer = optim.Adagrad(model.parameters(), lr=lr)
    if print_mat:
        print('Initial weigths of the matrix:')
        print(list(model.parameters()))

    train_loss = []

    for epoch in range(n_epoch):
        loss,_ = train_model(model, train_x, train_y, loss_fn, optimizer, batch_size)
        train_loss.append(loss)
        if print_feat:
            if epoch%n_print == 0:
                print('\n\n\nEpoch:', epoch+1, 'train_loss:', loss)
                print('weight:')
                print(list(model.parameters()))
    if print_mat:
        print('Final weigths of the matrix:')
        print(list(model.parameters()))
    weights = model.state_dict()
    lin_weight = weights['lin_ode.weight']
    nl_weight = weights['nl_ode.weight']
    if print_feat:
        plt.plot(train_loss)
        plt.show()
    return train_loss, lin_weight, nl_weight

def prediction(lin_weight, nl_weight, sol, T):
    lin_weight = np.array(lin_weight)
    nl_weight = np.array(nl_weight)
    for i in range(100):
        x = np.asarray(sol[-1])
        Ax = lin_weight.dot(x)

        phix = x.T.dot(nl_weight).dot(x)

        f = Ax + phix
        sol.append(f)
    return np.array(sol)
