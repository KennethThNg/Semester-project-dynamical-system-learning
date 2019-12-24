import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from grid_search import *
from models import *

data_name = 'pendulum'

if data_name == 'pendulum':
    print('generate pendulum data')
    k = 0.01
    sol = [[0.99,1]]
    dt = 1
    T = 100
    train_x, train_y, H = generate_pendulum(k, sol, dt, T)

    B = torch.zeros(2,2,2)
    z_dim = 2
    data = 'pend_df'

elif data_name == 'lorenz':
    print('generate lorenz data:')

    x = 0
    y = 1
    z = 1.05
    sigma = 10
    rho = 30
    beta = 8/3
    dt = 0.01

    sol = [[x, y, z]]
    z_dim = 3
    data = 'lorenz_df'

    train_x, train_y, H, B = generate_lorenz(sigma, rho, beta, sol)
else:
    raise ValueError('Unknown data')

print('train_x size:', train_x.size())
print('train_y size:', train_y.size())

print('Create lists of hyperparameters')
n_sim = 40
epoch = [200, 500, 1000]
batch = [10, 50, 100]
noise=[0.,0.1,0.2,0.3,0.4,0.5]
T = 100 #or 1000

print('Init the model')
model = NNODEModel(z_dim, z_dim, z_dim)

print('grid search')
grid_val = grid_search(model, train_x, train_y, T, z_dim, H, B, n_sim, noise, epoch, batch)

print('Create dataframe')
grid_df = pd.DataFrame(grid_val)
path_gen = '../gen_data/' #Create a file to store the csv
n_sim_str = '_' + str(T)
grid_df.to_csv(path_gen + data + n_sim_str + '.csv')
