from utiils import *
import torch
import matplotlib.pyplot as plt
import seaborn as sns

def grid_search(model, train_x, train_y, T, z_dim, H, B, n_sim=40, noise=[0.,0.1,0.2,0.3,0.4,0.5], epoch=[200, 500, 1000], batch_size=[10, 50, 100], lr=1e-2):
    '''

    :param model (nn.Module): neural network model
    :param train_x (torch.Tensor): training input
    :param train_y (torch.Tensor): training target
    :param T (int): time length
    :param z_dim (int): input dimension
    :param H (torch.Tensor): weight tensor of dimension [z_dim, z_dim]
    :param B (torch.Tensor): weight tensor of dimension [z_dim, z_dim, z_dim]
    :param n_sim (int): number of simulation
    :param noise (list): noise variances (float)
    :param epoch (list): list of epoch (int)
    :param batch_size (list): list of batch size (int)
    :param lr (float):
    :return:
    results (dict): contains the lists and the accuracy measurements (min_loss, lin_diff, quad_diff)
    '''
    results = []
    print('start simulation')
    #simulate the model n_sim time
    for s in range(n_sim):
        print('simulation:', s+1)
        #iterate on each element of the lists
        for std in noise:
            for e in epoch:
                for b in batch_size:
                    torch.manual_seed(s)
                    #add noise to data
                    norm_noise = torch.randn(T, z_dim)*std
                    train_x_noise = train_x + norm_noise
                    train_y_noise = train_y + norm_noise

                    #train the model and store the results
                    train_loss, _, _ = training_session(model, train_x_noise, train_y_noise, lr, 10, e, b)
                    weights = model.state_dict()
                    lin_weights = weights['lin_ode.weight'].detach().numpy().copy()
                    norm_lin = tensor_norm(H - weights['lin_ode.weight']).detach().numpy().copy()
                    #print(lin_weights)
                    nl_weights = weights['nl_ode.weight'].detach().numpy().copy()
                    norm_nl = tensor_norm(B - weights['nl_ode.weight']).detach().numpy().copy()
                    min_loss = train_loss[e-1]
                    #print(min_loss)

                    cross = {
                        'noise_std': std,
                        'epoch': e,
                        'batch': b,
                        'min_loss': min_loss,
                        'Linear diff': norm_lin,
                        'Quadratic diff': norm_nl,
                        'Linear': lin_weights,
                        'Quadratic': nl_weights
                    }
                    #print(cross['Linear'])
                    #print('-------------------------------')
                    results.append(cross)
                    model.reset_parameters()
                    #print('reset')
                    #print(lin_weights)
                    #print('-------------------------------')
    print('end simulation')
    return results

def boxplot_feature(data, fixed_feat1, val_feat1, fixed_feat2, val_feat2, feat_x, feat_y='min_loss', save_fig=False,
                    name='pend_box', data_name='pendulum'):
    '''
    Boxplot of the simulation
    :param data (pd.DataFrame):
    :param fixed_feat1 (str):
    :param val_feat1 (int or float):
    :param fixed_feat2 (str):
    :param val_feat2 (int or float):
    :param feat_x (str):
    :param feat_y (str):
    :param save_fig (bool):
    :param name (str):
    :param data_name (str):
    :return:
    boxplot of the data features
    '''
    data_feat = data[(data[fixed_feat1] == val_feat1)&(data[fixed_feat2] == val_feat2)]
    plt.figure(figsize=(20,10))
    sns.set(font_scale=4)
    ax = sns.boxplot(x=feat_x, y=feat_y, data=data_feat)
    plt.title(data_name + ' boxplot')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if save_fig:
        plt.savefig('../figure/boxplot/' + name + '.png')
