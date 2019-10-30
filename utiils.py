import torch

def batch_sampling(obs, times, t_max, batch_size = 100, delta=50):
    obs_ = torch.zeros((batch_size, delta, obs.shape[1], obs.shape[2]))
    ts_ = torch.zeros((batch_size, delta, times.shape[1], times.shape[2]))

    for b in range(batch_size):
        t0 = np.random.uniform(0, float(t_max))
        t1 = float(t0) + delta
        idx = np.arange(t0,t1)
        if len(idx) == delta:
            obs_[b,:,:] = obs[idx]
            ts_[b,:,:] = times[idx]

    return obs_, ts_
