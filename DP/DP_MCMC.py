import numpy as np
import scipy.stats as stats
import pandas as pd
from itertools import compress
import matplotlib.pyplot as plt
from sklearn import mixture

def CRP(alpha, N):
    Nc = {1:1}
    current_tbl = 1
    tbl = [1]
    C = [1]

    for i in range(2,N+1):
        cluster_prob_list = []
        for tbl_idx in tbl:
            cluster_prob_list.append( Nc[tbl_idx] / (i-1+alpha) )
        cluster_prob_list.append( alpha / (i-1 + alpha)  )
        ci = np.random.choice( list(range(1,len(cluster_prob_list)+1)), 1,
                               p=cluster_prob_list  )[0]
        if ci not in tbl:
            tbl.append(ci)
            Nc[ci] = 1
        else:
            Nc[ci] += 1
        C.append(ci)
    return C, Nc, tbl

def clearup_cls(tbl,Nc,C):
    Nc_temp = Nc
    for idx, keyval in enumerate(Nc.keys()):
        if Nc[keyval] == 0:
            tbl.remove(keyval)
            Nc_temp.pop(keyval)
    Nc = Nc_temp
    return tbl,Nc

def param_gen(tbl):
    lambda_dict = {}
    U_dict = {}
    mu_dict = {}

    for idx in range(1, len(tbl) + 1):
        lambda_elem = np.random.gamma(1, 1, size=1)[0]
        U_elem = np.random.uniform(0, 100, 1)[0]
        mu_elem = np.random.normal(0, 1 / (lambda_elem * U_elem))

        lambda_dict[idx] = lambda_elem
        U_dict[idx] = U_elem
        mu_dict[idx] = mu_elem
        params = [lambda_dict, U_dict, mu_dict]
    return params

def cls_sample(alpha,xi, params, tbl, Nc, C):
    # number of distinct clusters
    k_ = len(tbl)

    lambda_dict, U_dict, mu_dict = params

    # Sample parameter for k_ + 1 cls
    lambda_new = np.random.gamma(1, 1, size=1)[0]
    U_new = np.random.uniform(0, 10, 1)[0]
    mu_new = np.random.normal(0, 1 / (lambda_new * U_new))

    prob_list = []
    for tbl_idx in tbl: # up to k_
        prob_val = stats.norm.pdf(xi, mu_dict[tbl_idx],
                                     1 / (lambda_dict[tbl_idx] * U_dict[tbl_idx])
                                     )
        prob_list.append(prob_val * Nc[tbl_idx])

    # for k_ + 1
    prob_val = stats.norm.pdf(xi, mu_new,
                                 1 / (lambda_new * U_new)
                                 )
    prob_list.append( (alpha / (k_ + 1)) * prob_val)
    prob_list /= sum(prob_list)

    ci = np.random.choice(tbl + [k_ + 1], 1, p=prob_list)[0]
    if ci not in tbl:
        lambda_dict[ci] = lambda_new
        U_dict[ci] = U_new
        mu_dict[ci] = mu_new
    params = [lambda_dict, U_dict, mu_dict]
    return ci, params

def arrange_cls(tbl,Nc,C, params, idx, c_new):
    lambda_dict, U_dict, mu_dict = params

    if c_new not in tbl:
        tbl.append(c_new)
        Nc[c_new] = 1
        lambda_new = np.random.gamma(1, 1, size=1)[0]
        U_new = np.random.uniform(0, 10, 1)[0]
        mu_new = np.random.normal(0, 1 / (lambda_new * U_new))

        lambda_dict[c_new] = lambda_new
        U_dict[c_new] = U_new
        mu_dict[c_new] = mu_new

    else:
        Nc[c_new] += 1
    c_old = C[idx]
    C[idx] = c_new
    Nc[c_old] -= 1

    if Nc[c_old] == 0:
        tbl.remove(c_old)
        lambda_dict.pop(c_old)
        U_dict.pop(c_old)
        mu_dict.pop(c_old)

    params = [lambda_dict, U_dict, mu_dict]
    return tbl,Nc, C, params




def final_arrange_cls(tbl,Nc,C, params):
    lambda_dict, U_dict, mu_dict = params

    lambda_dict_new = dict()
    U_dict_new = dict()
    mu_dict_new = dict()

    num_cls = len(tbl)
    new_tbl = list(range(1,len(tbl)+1))
    Nc_new = dict()
    for idx in range(1,num_cls+1):
        Nc_new[idx] = 0

    Arr_dict = dict() # Change original cls to new cls
    for idx,keyval in enumerate(tbl):
        Arr_dict[keyval] = idx+1

    for idx in range(len(C)):
        C[idx] = Arr_dict[C[idx]]

    for clsidx in tbl:
        mat_cls = Arr_dict[clsidx]
        Nc_new[mat_cls] = Nc[clsidx]

        lambda_dict_new[mat_cls] = lambda_dict[clsidx]
        U_dict_new[mat_cls] = U_dict[clsidx]
        mu_dict_new[mat_cls] = mu_dict[clsidx]


    Nc_temp = Nc_new
    for idx, keyval in enumerate(Nc_new.keys()):
        if Nc_new[keyval] == 0:
            tbl.remove(keyval)
            Nc_temp.pop(keyval)
            lambda_dict_new.pop(keyval)
            U_dict_new.pop(keyval)
            mu_dict_new.pop(keyval)

    Nc_new = Nc_temp
    params = [lambda_dict_new, U_dict_new, mu_dict_new]
    tbl = new_tbl
    Nc = Nc_new

    return tbl,Nc,C, params


def Data_generation(num_cls, weights, num_data):
    mus = np.random.normal(0, 30, num_cls)
    sigs = np.random.uniform(0.5,1.5, num_cls)

    X = list()
    true_C = list()
    for idx in range(num_data):
        cls_idx = np.random.choice(list(range(0,num_cls)),1,weights)
        true_C.append(cls_idx)
        mu = mus[cls_idx]
        sig = sigs[cls_idx]
        xi = np.random.normal(mu,sig,1)[0]
        X.append(xi)
    return X, true_C, mus, sigs

def jump_sample(start_val,dim=3):
    jump_val = []
    for idx in range(dim):
        jump_val.append( np.random.normal(loc=start_val[idx],scale=1) )
    return jump_val

def compute_logpdf(obs, params):
    lam,U,mu = params
    # Compute PDF
    log_prob = 0
    for idx in range(len(obs)):
        xi = obs[idx]
        log_prob += np.log(stats.norm.pdf(xi, mu, 1 / (lam * U)))
    log_prob += np.log(stats.norm.pdf(mu, 0, 1 / (lam * U)))
    log_prob += stats.gamma.logpdf(lam, 1)
    return log_prob

def MH_update(obs, params,cls_idx, num_iter):
    lambda_dict, U_dict, mu_dict = params
    lam0 = lambda_dict[cls_idx]
    U0 = U_dict[cls_idx]
    mu0 = mu_dict[cls_idx]
    # Initial value setting
    x0 = [lam0,U0,mu0]
    param_list = [x0]

    for iteridx in range(num_iter):
        x_ = jump_sample(x0)

        f0 = compute_logpdf(obs,x0)
        f_ = compute_logpdf(obs,x_)

        alpha = np.exp( f_ - f0 )
        # alpha = np.exp(f_) / np.exp(f0)
        bouncer = np.random.random()
        # print(alpha, bouncer)
        if bouncer <= alpha or alpha == np.nan:
            param_list.append(x_)
            x0 = x_
        else:
            param_list.append(x0)
    return param_list











# Initial setup
K = 6
weights = [0.3,0.2,0.1,0.1,0.2,0.1]
N = 5000

X, true_C, true_mus, true_sigs = Data_generation(K,weights,N)
print(true_mus)
print(true_sigs)
print('')

# Initialization
alpha = 2
C,Nc,tbl = CRP(alpha, N)
tbl,Nc = clearup_cls(tbl,Nc,C)
params = param_gen(tbl)

dpgmm =  mixture.BayesianGaussianMixture(
    n_components=20, weight_concentration_prior=1 / N, max_iter= 1000).fit(np.matrix(X).T)

print(tbl)
print(Nc)
print('')

num_trial = 500
for trial in range(num_trial):
    print(trial)
    for idx in range(N):
        k_ = len(tbl)
        xi = X[idx]
        if Nc[C[idx]] == 1 : # ci != cj for all j != i,
            if np.random.binomial(n=1, p= k_/(k_+1)) == 1:
                continue # do nothing
            else:
                c_new = k_ + 1
                tbl, Nc, C, params = arrange_cls(tbl, Nc, C, params, idx, c_new)
        else: # if ci = cj for some j != i,
            c_new, params = cls_sample(alpha, xi, params, tbl, Nc, C)
            tbl, Nc, C, params = arrange_cls(tbl, Nc, C, params, idx, c_new)

    tbl, Nc, C, params = final_arrange_cls(tbl,Nc,C, params)

    # parameter update
    num_iter = 500
    burnup = 400
    lambda_dict, U_dict, mu_dict = params
    for cls_idx in tbl:
        X_cls = [X[i] for i in range(len(C)) if C[i] == cls_idx ]
        param_list = MH_update(X_cls,params,cls_idx, num_iter)
        param_list = param_list[burnup:]
        param_list = np.sum(param_list,axis=0) / (num_iter - burnup)

        lambda_dict[cls_idx] = param_list[0]
        U_dict[cls_idx] = param_list[1]
        mu_dict[cls_idx] = param_list[2]

    params = [lambda_dict, U_dict, mu_dict]
    print(tbl)
    print(Nc)
    print(mu_dict)
    print('')

    # trial += 1