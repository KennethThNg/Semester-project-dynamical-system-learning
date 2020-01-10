import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
#----------------------------------------------
##DATA GENERATION
def generate_pendulum(k, sol, dt=1, T=100, dim=2):
    '''
    Generate the data from the dynamical system of the pendulum motion
    :param k (int): spring constant
    :param sol (np.array): initial condition
    :param dt (float): time-step
    :param T (int): time length
    :param dim (int): data dimension
    :return:
    train_x (torch.Tensor): training input of dimension [T, dim]
    train_y (torch.Tensor): training target of dimentions [T, dim]
    torch.Tensor(AA): weight matrix of the dynamical system of dimension [dim, dim]
    '''
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
    '''

    :param sigma (float): constant
    :param rho (float): chaotic constant
    :param beta (float): constant
    :param sol (numpy.array): initial condition
    :param dt (float): time-step
    :param T (int): time length
    :param dim (int): data dimension
    :return:
    train_x (torch.Tensor): training input of dimension [T, dim]
    train_y (torch.Tensor): training target of dimentions [T, dim]
    torch.Tensor(AA): weight matrix of the dynamical system of dimension [dim, dim]
    torch.Tensor(B): weight tensor of the dynamical system of dimension [dim, dim, dim]
    '''
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

def create_matrix_time(tensor, N):
    '''
    Create multiple time series of length N from one time serie.
    :param tensor (torch.Tensor): tensor of shape [time length, 1]
    :param N (int): time length of each time series
    :return:
    new_tensor (torch.Tensor): tensor containing all time series of length N.
    '''
    new_tensor = torch.zeros(N, tensor.size(0) - N + 1)
    for k in range(new_tensor.size(1)):
        new_tensor[:,k] = tensor[k:(k + N)].squeeze()
    return new_tensor

#----------------------------------------------
## TENSOR NORM

def tensor_norm(tensor):
    '''
    Compute the norm of a tensor of type torch.Tensor
    :param tensor (torch.Tensor): d-dim tensor (d=2 or 3)
    :return:
    norm of the tensor of type torch.Tensor of length 1
    '''
    a = tensor.view(-1)
    b = torch.norm(a, p=2)
    return b

#----------------------------------------------
## TRAINING THE MODEL
def train_test_split(feature, ratio=0.8):
    '''
    Split the dataset into training and train set and test set
    :param feature (numpy.array): dataset of shape [time length, dimension]
    :param ratio (int): split ratio
    :return:
    train_set (numpy.array): training set of shape [train_size, dimension]
    test_set (numpy.array): test set of shape [test_size, dimension]
    '''
    train_size = int(len(feature)*ratio)
    train_set = feature[:train_size]
    test_set = feature[train_size:]
    return train_set, test_set

def input_target_split(feature, delta, N):
    '''
    Split the dataset into input and target set
    :param feature (torch.Tensor): tensor to split of shape [time length, dimnension]
    :param delta (int): gap between the target and the input
    :param N (int): number of time series in the matrix
    :return:
    matrix_input (torch.Tensor) input set of shape [time length - delta +, N, dimnesion]
    matrix_target (torch.Tensor) target set of shape [time length - delta +, N, dimnesion]
    '''
    input_ = torch.FloatTensor(feature[:-delta])
    target = torch.FloatTensor(feature[delta:])
    matrix_input = create_matrix_time(input_, N).permute(1,0)
    matrix_target = create_matrix_time(target, N).permute(1,0)
    return matrix_input, matrix_target

def run_session(model, optimizer, train_x, train_y, test_x, test_y, n_epoch=1000, batch=10):
    '''
    Train the model by updating its weights paraemters
    :param model (nn.Module): model to train
    :param optimizer (nn.Optimizer): optimizer use to update the model
    :param train_x (torch.Tensor): train input of shape [number time series, time legnth]
    :param train_y (torch.Tensor): train target of shape [number of time series, time length]
    :param test_x (torch.Tensor) : test input of shape [number time series, time legnth]
    :param test_y (torch.Tensor): test target of shape [number time series, time legnth]
    :param n_epoch (int): number of epochs
    :param batch (int): number of sample per batch
    :return:
    model trained with optimal weights
    '''
    loss_fn = nn.MSELoss()
    for epoch in range(n_epoch):
        model.train()
        train_loss,_ = train_model(model, train_x, train_y, loss_fn, optimizer, batch_size=batch)
        model.eval()
        test_loss,_ = test_model(model, test_x, test_y, loss_fn, batch_size=batch)
        if epoch%10 == 0:
            print('Epoch:', epoch+1, ', train_loss:', train_loss, ', test_loss:', test_loss)

def train_model(model, train_x, train_y, loss_fn, optimizer, batch_size):
    '''
    Train the model and compute the training loss and the prediction of the model
    :param model (nn.Module): neural net to train
    :param train_x (torch.Tensor): training input
    :param train_y (torch.Tensor): training target
    :param loss_fn (nn.Module): loss function
    :param optimizer (nn.Optimizer): learning algorithm
    :param batch_size (Int): amount of data in the selected batch
    :return:
    train_loss (float): training loss of the model
    pred (torch.Tensor): prediction of the model after training
    '''
    #init train_loss
    train_loss = 0

    #Train per batch
    for b in range(0, train_x.size(0), batch_size):
        #zero the gradient
        model.zero_grad()

        #Forward
        pred = model(train_x.narrow(0, b, batch_size))
        loss = loss_fn(pred, train_y.narrow(0, b, batch_size))

        #Store the loss
        train_loss += loss.item()

        #backward
        loss.backward()

        #update of the optimizer
        optimizer.step()
    return train_loss, pred

def test_model(model, test_x, test_y, loss_fn, batch_size=1):
    '''
    Test and validate the model
    :param model (nn.Module): neural net to validate
    :param test_x (torch.Tensor): test input
    :param test_y (torch.Tensor): test target
    :param loss_fn (nn.Module): loss function
    :param batch_size (int): amount of data in the selected batch
    :return:
    test_loss (float): test loss of the model
    pred (torch.Tensor): prediction of the model after training
    '''
    #init test loss
    test_loss = 0

    #unable backward
    with torch.no_grad():
        for b in range(0, test_x.size(0), batch_size):
            #Forward
            pred = model(test_x.narrow(0, b, batch_size))
            loss = loss_fn(pred, test_y.narrow(0, b, batch_size))

            #Store loss
            test_loss += loss.item()
        return test_loss, pred

def training_session(model, train_x, train_y, lr=1e-2, n_print=10, n_epoch=100, batch_size=1, print_mat=False, print_feat=False):
    '''
    train the model and return the training loss and the final weights and plot the results.
    This method is used for the model simulation with noisy data
    :param model (nn.Module): neural net to train
    :param train_x (torch.Tensor): training input
    :param train_y (torch.Tensor): training target
    :param lr (float): learning rate
    :param n_print (int): print the results after n_print epochs
    :param n_epoch (int): number of iterations
    :param batch_size (int): number of date per batch
    :param print_mat (bool): determine if the results are printed
    :param print_feat (bool): determine if the weights are stored.
    :return:
    train_loss (float): training loss
    lin_weight (torch.Tensor): 2-dim weight obtained after training
    nl_weight (torch.Tensor): 3-dim weight obtained after training
    '''

    #Create the loss and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.Adagrad(model.parameters(), lr=lr)
    if print_mat:
        print('Initial weigths of the matrix:')
        print(list(model.parameters()))

    #list of loss
    train_loss = []

    #start training
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

    #store the weights
    weights = model.state_dict()
    lin_weight = weights['lin_ode.weight']
    nl_weight = weights['nl_ode.weight']

    #plot the results
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
