import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from Causal_bound import Causal_bound
from Causal_bound import causal_bound_case1
from Data_generation import Data_generation

def intfy(W):
    return np.array(list(map(int,W)))

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

# Setting
T = 5000 # number of rounds
case_num = 4
if case_num != 4:
    dim = 1 # Dimension. Only matter for case_num = 4
else:
    dim = 100

N = 100*dim
# N = max(T*100,dim*200) # Number of data to be sampled (for accurate interventional effect)
Ns = 20*dim # Number of Ds

Obs, Intv, Intv_S =  Data_generation(case_num=case_num ,N=N,Ns=Ns,dim = dim,seed_num=12345)

# Identify optimal arm, optimal rewards, and rewards for all arm.
## Binary {0,1} are assumed, but one can generalize.
opt_arm, opt_reward, rewards = true_answer(Intv=Intv)

# If not case 1, (Y continuous)
if case_num != 1:
    std_sydox0 = np.std(Intv_S[Intv_S['X'] == 0]['Y'])
    std_sydox1 = np.std(Intv_S[Intv_S['X'] == 1]['Y'])
    std_sydox = [std_sydox0, std_sydox1]

    std_yx0 = np.std(Obs[Obs['X']==0]['Y'])
    std_yx1 = np.std(Obs[Obs['X'] == 1]['Y'])
    std_yx = [std_yx0, std_yx1]

# Compute LB, UB using causal bound technique.
## Not necessary for KB-UCB.
LB0, p_true0, UB0 = Causal_bound(case_num=case_num , dim=dim, Obs=Obs, Intv=Intv, Intv_S = Intv_S, x=0, z=0)
LB1, p_true1, UB1 = Causal_bound(case_num=case_num , dim=dim, Obs=Obs, Intv=Intv, Intv_S = Intv_S, x=1, z=0)
LB = [LB0,LB1]
L_max = np.max(LB)
UB = [UB0,UB1]

K = 2 # Number of arms.
Arm = []
Reward = []

# Number of selection of arm a at time t
## if we choose a at time t, then Na_T +=1 from t+1
Na_T = dict()
sum_reward = 0 # sum of rewards by far to compute cum.regret.

''' Bandit start! '''
# Initialization
for t in range(K):
    at = t # For each arm
    Na_T[at] = 0 # Setting each arm selection to 0
    rt = pulling_receive(at,Intv,Na_T) # Pulling and receive rewards
    Arm.append(at)
    Reward.append(rt)

    # Last step for each round
    ## Update the pulled arm selection
    Na_T[at] += 1

    ## Update regret
    ### Choose mean of rewards for arm at
    sum_reward += rewards[at]
    # As t starts from 0, we are at t+1 rounds.
    cum_regret = (t+1)*opt_reward - sum_reward

# Initializing the probability of choosing arm
prob_opt = Na_T[opt_arm] / K
Bandit = pd.DataFrame({'Arm':Arm, 'Reward':Reward, 'Cum_regret': cum_regret, 'Prob_opt':prob_opt })
ft = lambda x: np.log(x) + 3*np.log( np.log( x ) )

# Iterate for round K to T
for t in range(K, T):
    # For each round, we compute upper bound for each arm,
    # and select one with highest value.
    Ua_t_list = []
    for a in range(K): # for each arm,
        # Compute the empirical mean for each arm by far.
        mu_hat = np.mean(Bandit[Bandit['Arm'] == a]['Reward'])
        # Compute the KL_bound
        ubKL = ft(t) / Na_T[a]
        UB_a = UB[a]

        # Compute upper bound Ua(t) for arm a at time t
        ## if Y are binary
        if case_num == 1:
            Ua_t = upper_bound_KL_case1(p=mu_hat,ubKL=ubKL)
            if pd.isnull(UB_a):
                Ua_t_list.append(Ua_t)
            else:
                Ua_t_list.append(min(Ua_t, UB_a))
        ## If Y are continuous
        elif case_num != 1:
            # If we don't have enough data to estimate sigma_do
            # Compute using Ds
            # Otherwise, Compute using accumulated data
            if Na_T[a] < Ns: # If Ds is more accurate
                # From Intv_S sample
                std_ydox_a = std_sydox[a]
                # From accumulate bandit
                std_yhat_a = np.std(Bandit[Bandit['Arm']==a]['Reward'])
            else:
                # From Bandit for both
                std_ydox_a = np.std(Bandit[Bandit['Arm'] == a]['Reward'])
                std_yhat_a = np.std(Bandit[Bandit['Arm'] == a]['Reward'])

            Ua_t = upper_bound_KL_case3(mu_hat=mu_hat, ubKL=ubKL,std_sydox = std_ydox_a,std_hat = std_yhat_a)
            if pd.isnull(Ua_t):
                Ua_t_list.append(UB_a)
            else:
                Ua_t_list.append(min(Ua_t, UB_a ) )

            # print(t,a,UB_a, Ua_t, min(Ua_t, UB_a ))
    # Choose arm with maximal UB
    at= np.argmax(Ua_t_list)
    if UB_a < Ua_t and opt_arm == at:
        print(at,"works!")
    rt = pulling_receive(at,Intv,Na_T)

    # Add the visiting number after pulling
    Na_T[at] += 1

    # We are at t+1 rounds.
    prob_opt = Na_T[opt_arm] / (t + 1 )
    sum_reward += rewards[int(at)]
    cum_regret = round((t+1) * opt_reward - sum_reward,3)

    Bandit_iter = pd.DataFrame({'Arm': [at], 'Reward': [rt], 'Cum_regret': cum_regret,'Prob_opt':prob_opt})
    Bandit_iter.index = [t]
    Bandit = Bandit.append(Bandit_iter)

Bandit_BKL = Bandit

# # Graphical illustration
# f, ax = plt.subplots(2, sharex='all')
# plt.title("B-KL-UCB")
# ax[0].set_title('Cumulative regret')
# ax[0].plot(Bandit['Cum_regret'])
#
# ax[1].set_title('Probability of choosing opt arm')
# ax[1].plot(Bandit['Prob_opt'])

