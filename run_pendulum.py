from models import *
from utiils import *
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

print('generate pendulum data')

k = 0.01
sol = [[0.99,1]]
dt = 1
T = 100
train_x, train_y, H = generate_pendulum(k, sol, dt, T)
print('train_x.shape:', train_x.shape)#train_input
print('train_y.shape:', train_y.shape)#train_output

plt.figure(1)
plt.plot(train_x[:,0].numpy());
plt.plot(train_y[:,0].numpy())
plt.title('Pendulum')

model_name = 'simple_pendulum'

if model_name == 'simple_pendulum':
    print('training pendulum with simple linear model:')
    model = LinearODEModel(2, 2)
    bs = train_x.size(0)
    T = bs
    n_epoch = 600
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
elif model_name == 'pendulum_bias':
    print('training pendulum model with bias:')
    model = LinearODEModel(2, 2, True)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    bs = train_x.size(0)
    T = bs
    n_epoch = 600
elif model_name == 'pendulum_batch_10':
    print('training pendulum model with batch size 10:')
    model = LinearODEModel(2, 2)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    bs = 10
    T = train_x.size(0)
    n_epoch = 600
elif model_name == 'non_lin_pendulum':
    model = NNODEModel(2,2,2)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adagrad(model.parameters(), lr=1e-2)
    bs = 10
    T = 30
    n_epoch = 600
else:
    raise ValueError('Unknown model')

print('Real weigths of the matrix:')
print(H)
print('Initial weigths of the matrix:')
print(list(model.parameters()))

train_losses = []

for epoch in range(n_epoch):  # number of epochs
    train_loss, prediction = train_model(model, train_x, train_y, loss_fn, optimizer, batch_size=bs)
    train_losses.append(train_loss)


print('Final weigths of the matrix:')
print(list(model.parameters()))
plt.figure(2)
plt.plot(train_losses, label='train loss')
plt.title('Train loss of pendulum')

plt.figure(3)
plt.plot(train_y[0:T,0].numpy(), label='Ground truth')
plt.plot(prediction[0:T,0].detach().numpy(), label='Prediction')
plt.title('Prediction')
plt.legend()

plt.show()
