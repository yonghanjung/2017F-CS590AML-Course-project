import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from Causal_bound import Causal_bound
from Causal_bound import causal_bound_case1
from sklearn import linear_model
from Data_generation import Data_generation

def intfy(W):
    return np.array(list(map(int,W)))

def linear_training(X,Y):
    clf = linear_model.LinearRegression()
    clf.fit(X, Y)
    return clf

def logistic_training(X,Y):
    clf = linear_model.LogisticRegression()
    clf.fit(X,Y)
    return clf

def inverse_logit(Z):
    return np.exp(Z) / (np.exp(Z)+1)


def upper_bound_KL_case1(p, ubKL):
    q = causal_bound_case1(p,ubKL,'max')
    return q

def upper_bound_KL_case3(mu_hat, ubKL, std_sydox, std_hat):
    # In here, std_ymux is computed from Ds
    if std_sydox != std_hat:
        C = ubKL + 0.5 - np.log(std_sydox / std_hat) - (std_hat ** 2) / (2 * std_sydox ** 2)
    else:
        C = ubKL + 0.5 - (std_hat ** 2) / (2 * std_sydox ** 2)
    UB = mu_hat + std_sydox * np.sqrt(2 * C)
    return UB

def true_answer(Intv):
    reward_0 = np.mean( Intv[Intv['X']==0]['Y'] )
    reward_1 = np.mean(Intv[Intv['X'] == 1]['Y'])
    rewards = [reward_0, reward_1]
    return np.argmax(rewards), np.max(rewards), rewards

def pulling_receive(at,Intv, Na_T):
    return list(Intv[Intv['X'] == at]['Y'])[Na_T[at]]

def Baseline_UCB(case_num, alpha, Psi_t, Obs, dim, K, t):
    if case_num == 31:
        A = np.matrix(np.eye(dim))
        b = np.zeros((dim,1))
        z = Obs['Z'].loc[t]
        width = []
        rewards = []
        UBs = []

        if len(Psi_t) > 1:
            for idx in Psi_t[1:]:
                x_ta = np.matrix( Obs[['X','Z']].loc[idx] )
                A += x_ta.T * x_ta
                b += np.matrix( Obs['Y'].loc[idx] * x_ta ).transpose()

            theta = A.I * b
            for a in range(K):
                x_ta = np.matrix([a,z])
                s_ta = np.array(x_ta * A.I * x_ta.T)[0][0]
                w_ta = alpha * s_ta
                r_ta = np.array(x_ta * theta)[0][0]
                width.append(w_ta)
                rewards.append(r_ta)
                UBs.append(w_ta + r_ta)
            return width, rewards, UBs
        else:
            theta = A.I * b
            for a in range(K):
                x_ta = np.matrix([a, z])
                s_ta = np.array(x_ta * A.I * x_ta.T)[0][0]
                w_ta = alpha * s_ta
                r_ta = np.array(x_ta * theta)[0][0]
                width.append(w_ta)
                rewards.append(r_ta)
                UBs.append(w_ta + r_ta)
            return width, rewards, UBs

# Setting
T = 5000 # number of rounds
case_num = 41
if case_num != 4 and case_num != 41:
    dim = 1 # Dimension. Only matter for case_num = 4
else:
    dim = 100

mat_dim = dim+1 #including arm
N = 10000
# N = max(T*100,dim*200) # Number of data to be sampled (for accurate interventional effect)
Ns = int(N/10) # Number of Ds

Obs, Intv, Intv_S = Data_generation(case_num=case_num, N=N, Ns=Ns, dim=dim, seed_num=2017)

obs_fit = linear_training(Obs[['X'] + list(range(dim)) ],Obs['Y'])
obs_coef = obs_fit.coef_
obs_result = obs_fit.predict(Obs[['X'] + list(range(dim)) ])
obs_resid = Obs['Y'] - obs_result

obs_x_fit = logistic_training(Obs[list(range(dim))], Obs['X'])


true_fit = linear_training( Intv[['X'] + list(range(dim)) ],Intv['Y'] )
true_coef = true_fit.coef_
true_coef = np.matrix(true_coef).T
true_result = true_fit.predict(Intv[['X'] + list(range(dim)) ])
true_resid = Intv['Y'] - true_result

s_true_fit = linear_training( Intv_S[['X'] + list(range(dim)) ],Intv_S['Y'] )
s_true_coef = s_true_fit.coef_
s_true_result = s_true_fit.predict(Intv_S[['X'] + list(range(dim)) ])
s_true_resid = Intv_S['Y'] - s_true_result

std_do = np.std(true_resid)
std_obs = np.std(obs_resid)
std_s_do = np.std(s_true_resid)

K = 2 # Number of arms.
Arm = []
Reward = []
cum_regret = []
prob_opt = []
Na_T = dict()
for a in range(K):
    Na_T[a] = 0

lamb = 0.5
A = lamb * np.matrix( np.eye(mat_dim) )
b = np.matrix( np.zeros((mat_dim,1)) )

alpha = 3
flatten = lambda l: [item for sublist in l for item in sublist]


Bandit_BLin = {'Arm':Arm, 'Reward':Reward, 'Cum_regret': cum_regret, 'Prob_opt':prob_opt, 'UCB':[], 'UB':[] }
# Bandit_Lin = {'Arm':Arm, 'Reward':Reward}
sum_regret = 0

for t in range(T):
    theta_t = A.I * b

    # Observe Z
    Z = list( Intv[ list(range(dim)) ].loc[t] )
    UCB_t = []
    UB_t = []
    X_t = []
    true_rewards = []
    for a in range(K):
        x_ta = np.matrix( np.asarray( [a] + Z ) ).T
        w_ta = alpha * np.sqrt( np.asarray( x_ta.T * A.I * x_ta )[0][0] )
        ucb_ta = np.asarray(theta_t.T * x_ta)[0][0] + w_ta

        obs_mean = obs_fit.predict(np.array(flatten(x_ta.tolist())))[0]
        pxz = obs_x_fit.predict_proba(Z)[0][a]
        Hxz = -np.log(pxz)
        C = Hxz + 0.5 - np.log(std_s_do / std_obs) - (std_obs ** 2) / (2 * std_s_do ** 2)
        UB_a = obs_mean + std_s_do * np.sqrt(2 * C)

        true_rt = true_fit.predict(np.array(flatten(x_ta.tolist())))[0]
        true_rewards.append(true_rt)
        if pd.isnull(UB_a):
            UCB_t.append(ucb_ta)
        else:
            UCB_t.append( min(ucb_ta,UB_a ))
        X_t.append(x_ta)
        # UCB_t.append(ucb_ta)
        UB_t.append(UB_a)

    at = np.argmax(UCB_t)
    opt_arm = np.argmax(true_rewards)
    Na_T[at] += 1
    prob_opt = Na_T[opt_arm] / (t + 1)
    Bandit_BLin['Arm'].append(at)
    x_ta = X_t[at]

    # Observe reward
    rt = true_rewards[at]
    opt_reward = max(true_rewards)
    sum_regret += opt_reward - rt

    Bandit_BLin['Reward'].append(rt)
    Bandit_BLin['Cum_regret'].append(sum_regret)
    Bandit_BLin['Prob_opt'].append(prob_opt)
    Bandit_BLin['UCB'].append(UCB_t[at])
    Bandit_BLin['UB'].append(UB_t[at])
    b += rt * x_ta
    A += x_ta * x_ta.T
    print(round(t/T,3))

Bandit_BLin  = pd.DataFrame.from_dict(Bandit_BLin )
#
# # Graphical illustration
# f, ax = plt.subplots(2, sharex='all')
# ax[0].set_title('Cumulative regret')
# ax[0].plot(Bandit_BLin ['Cum_regret'])
#
# ax[1].set_title('Probability of choosing opt arm')
# ax[1].plot(Bandit_BLin ['Prob_opt'])