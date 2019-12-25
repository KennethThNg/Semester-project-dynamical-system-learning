from models import *
from utiils import *
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

print('generate lorenz data:')

x = 0
y = 1
z = 1.05
sigma = 10
rho = 30
beta = 8/3
dt = 0.01
T = 10000

sol = [[x, y, z]]

train_x, train_y, H, B = generate_lorenz(sigma, rho, beta, sol, dt, T)
print('Parameter:')
print(H)
print(B)
print('train_x size:', train_x.size())
print('train_y size:', train_y.size())

plt.figure(1)
fig = plt.figure(figsize=(20,10))
ax = fig.gca(projection='3d')
ax.plot(train_x[:,0].numpy(), train_x[:,1].numpy(), train_x[:,2].numpy())
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.title('Lorenz attractor')

model_name = 'non_linear'
if model_name == 'non_linear':
    print('training lorenz with simple non-linear model')
    model = NNODEModel(3,3,3)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adagrad(model.parameters(), lr=1e-2)
    bs = 100
elif model_name == 'non_linear_bias':
    print('training lorenz with bias')
    model = NNODEModel(3,3,3, True)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adagrad(model.parameters(), lr=1e-2)
    bs = 100
else:
    raise ValueError('Unknown model')

print('Initial weigths of the matrix:')
print(list(model.parameters()))
n_epoch = 10000
train_losses = []

for epoch in range(n_epoch):  # number of epochs
    train_loss, prediction = train_model(model, train_x, train_y, loss_fn, optimizer, batch_size=bs)
    train_losses.append(train_loss)

print('Final weigths of the matrix:')
print(list(model.parameters()))

plt.figure(2)
plt.plot(train_losses)
plt.title('Train loss of Lorenz')

plt.figure(3)
fig = plt.figure(figsize=(20,10))
ax = fig.gca(projection='3d')
ax.plot(train_x[:,0].numpy(), train_x[:,1].numpy(), train_x[:,2].numpy(), label='Ground truth');
ax.plot(prediction[:,0].detach().numpy(), prediction[:,1].detach().numpy(), prediction[:,2].detach().numpy(), label='Prediction')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.title('Prediction of Lorenz')
ax.legend()
plt.show()
