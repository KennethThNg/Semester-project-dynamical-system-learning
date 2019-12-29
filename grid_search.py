from utiils import *
import torch
import matplotlib.pyplot as plt
import seaborn as sns

def grid_search(model, train_x, train_y, T, z_dim, H, B, n_sim=40, noise=[0.,0.1,0.2,0.3,0.4,0.5], epoch=[200, 500, 1000], batch_size=[10, 50, 100], lr=1e-2):
    results = []
    print('start simulation')
    for s in range(n_sim):
        print('simulation:', s+1)
        for std in noise:
            for e in epoch:
                for b in batch_size:
                    torch.manual_seed(s)
                    norm_noise = torch.randn(T, z_dim)*std
                    train_x_noise = train_x + norm_noise
                    train_y_noise = train_y + norm_noise

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
    data_feat = data[(data[fixed_feat1] == val_feat1)&(data[fixed_feat2] == val_feat2)]
    plt.figure(figsize=(20,10))
    ax = sns.boxplot(x=feat_x, y=feat_y, data=data_feat)
    plt.title(data_name + 'boxplot')
    if save_fig:
        plt.savefig('../figure/boxplot'+ name + '.png')
