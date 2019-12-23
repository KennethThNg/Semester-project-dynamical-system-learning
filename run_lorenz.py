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

sol = [[x, y, z]]

train_x, train_y, H, B = generate_lorenz(sigma, rho, beta, sol)
print('Parameter:')
print(H)
print(B)
print('train_x size:', train_x.size())
print('train_y size:', train_y.size())

fig = plt.figure(figsize=(20,10))
ax = fig.gca(projection='3d')
ax.plot(train_x[:,0].numpy(), train_x[:,1].numpy(), train_x[:,2].numpy())
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

model_name = 'non_linear'
if model_name == 'non_linear':
    print('training lorenz with simple non-linear model')
elif model_name == 'non_linear_bias':
    print('training lorenz with bias')
else:
    raise ValueError('Unknown model')

plt.show()
