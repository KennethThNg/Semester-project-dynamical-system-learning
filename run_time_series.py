import torch
import torch.nn as nn
import torch.optim as optim
import Pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from models import NeuralNetModel
from utiils import train_model, test_model, create_matrix_time, train_test_split

print('load data:')
data_path = '../data/'
data = pd.read_parquet(data_path+'SPY_hourly.parquet')

plt.figure(1)
plt.plot(data.values)
plt.show()

print('Prepare data:')
#standardize
price = data.values
print('time series shape: ', price.shape)

print('Normalize data:')
price_centered = price - price.mean()
price_stand = price_centered/price.std()

print('split the data into train-test set:')
train_set, test_set = train_test_split(price_stand)
train_set_mis = train_set[:-5]
test_set_mis = test_set[:-6]

delta=5
print('create the input and the target with gap ' + str(delta))
train_input = torch.FloatTensor(train_set_mis[:-delta])
train_target = torch.FloatTensor(train_set_mis[delta:])
test_input = torch.FloatTensor(test_set_mis[:-delta])
test_target = torch.FloatTensor(test_set_mis[delta:])

N = 10
print('create tensor of multiple time series of length ' + str(N))
train_x = create_matrix_time(train_input, N).permute(1,0)
train_y = create_matrix_time(train_target, N).permute(1,0)
test_x = create_matrix_time(test_input, N).permute(1,0)
test_y = create_matrix_time(test_target, N).permute(1,0)
print('train input size: ', train_x.shape)
print('train target size: ', train_y.shape)
print('test input size: ', test_x.shape)
print('train input size: ', test_y.shape)

print('Create model')
in_dim = 10
out_dim = 10
hid_dim = 3
model = NeuralNetModel(input_dim=in_dim, hidden_dim=hid_dim, output_dim=out_dim)

loss_fn = nn.MSELoss()
optimizer = optim.Adagrad(model.parameters(), lr=1e-2)
n_epoch = 1000
train_losses = []
test_losses = []

print('start training...')
for epoch in range(n_epoch):
    model.train()
    train_loss,_ = train_model(model, train_x, train_y, loss_fn, optimizer, batch_size=10)
    train_losses.append(train_loss)
    model.eval()
    test_loss,_ = test_model(model, test_x, test_y, loss_fn, batch_size=10)
    test_losses.append(test_loss)
    if epoch%10 == 0:
        print('Epoch:', epoch+1, ', train_loss:', train_loss, ', test_loss:', test_loss)
print('...training finished')

print('prediction analysis:')
prediction = model(test_x)
#the time series in the frame
n=1

#time windows
k=10
window = 100
plt.figure(2)
plt.plot(test_y[k:(k+window),n].numpy(), label='ground truth');
plt.plot(prediction[k:(k+window),n].detach().numpy(), label='prediction')
plt.legend()
plt.title('prediction of the ' + str(n) +'-th time series')
plt.show()
