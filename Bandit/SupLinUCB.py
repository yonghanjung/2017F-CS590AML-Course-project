import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from Causal_bound import Causal_bound
from Causal_bound import causal_bound_case1
from sklearn import linear_model

def intfy(W):
    return np.array(list(map(int,W)))

def linear_training(X,Y):
    clf = linear_model.LinearRegression()
    clf.fit(X, Y)
    return clf

def inverse_logit(Z):
    return np.exp(Z) / (np.exp(Z)+1)

def Data_generation(case_num, N, Ns, dim, seed_num = 12345):
    if case_num == 1 or case_num == 11:
        np.random.seed(seed_num)

        U1 = np.random.normal(0, 5, N)
        U2 = np.random.normal(0, 5, N)
        U3 = np.random.binomial(1, 0.2, N)

        Z = U1 + U2
        X = np.round((inverse_logit(2 * U1 + 3 * Z) + U3) / 2, 0)
        Y = np.round((inverse_logit(2 * U2 - 3 * Z) + (X + U3) / 2) / 2, 0)

        X = intfy(X)
        Y = intfy(Y)

        X_intv = np.asarray([0] * int(N / 2) + [1] * int(N / 2))
        # Y_intv = X_intv * DZ * DU2 * DU3
        # X_intv = (X_intv+1)/2
        Y_intv = np.round((inverse_logit(2 * U2 - 3 * Z) + (X_intv + U3) / 2) / 2, 0)
        # Y_intv = (Y_intv + 1)/2
        X_intv = intfy(X_intv)
        Y_intv = intfy(Y_intv)

        X_obs = np.asarray(X)
        Y_obs = np.asarray(Y)
        Z_obs = np.asarray(Z)
        Obs = pd.DataFrame({'Z': Z_obs, 'X': X_obs, 'Y': Y_obs})

        X_intv = np.asarray(X_intv)
        Y_intv = np.asarray(Y_intv)
        Intv = pd.DataFrame({'X': X_intv, 'Y': Y_intv, 'Z': Z_obs})

        sample_indces_x1 = np.random.choice(list(range(0, int(N / 2))), int(Ns / 2), replace=False)
        sample_indces_x0 = np.random.choice(list(range(int(N / 2), N)), int(Ns / 2), replace=False)
        sample_indices = np.asarray(list(sample_indces_x1) + list(sample_indces_x0))

        X_sintv = X_intv[sample_indices]
        Y_sintv = Y_intv[sample_indices];
        Z_sintv = Z_obs[sample_indices]
        Intv_S = pd.DataFrame({'X': X_sintv, 'Y': Y_sintv, 'Z': Z_sintv})

        return Obs, Intv, Intv_S

    elif case_num == 3 or case_num == 31:
        np.random.seed(seed_num)
        std_var = 5
        U1 = np.random.normal(-6, std_var , N)
        U2 = np.random.normal(-10, std_var , N)
        U3 = np.random.normal(-3, std_var , N)

        Z = U1 - U2
        X = intfy(np.round(inverse_logit(1 * Z - 1 * U1 + 1 * U3), 0))
        Y = Z + U2 + U3 + std_var * 2 * X

        X_intv = intfy(np.asarray([0] * int(N / 2) + [1] * int(N / 2)))
        Y_intv = Z + U2 + U3 + std_var * 2 * X_intv

        X_obs = np.asarray(X)
        Y_obs = np.asarray(Y)
        Z_obs = np.asarray(Z)
        Obs = pd.DataFrame({'Z': Z_obs, 'X': X_obs, 'Y': Y_obs})

        X_intv = np.asarray(X_intv)
        Y_intv = np.asarray(Y_intv)
        Intv = pd.DataFrame({'X': X_intv, 'Y': Y_intv, 'Z': Z_obs})

        sample_indces_x1 = np.random.choice(list(range(0, int(N / 2))), int(Ns / 2), replace=False)
        sample_indces_x0 = np.random.choice(list(range(int(N / 2), N)), int(Ns / 2), replace=False)
        sample_indices = np.asarray(list(sample_indces_x1) + list(sample_indces_x0))

        X_sintv = X_intv[sample_indices]
        Y_sintv = Y_intv[sample_indices];
        Z_sintv = Z_obs[sample_indices]
        Intv_S = pd.DataFrame({'X': X_sintv, 'Y': Y_sintv, 'Z': Z_sintv})

        return Obs, Intv, Intv_S

    elif case_num == 4 or case_num == 41:
        np.random.seed(seed_num)
        mu1 = 2 * np.random.rand(dim) - 1
        mu2 = mu1 + 3 * np.random.rand(dim)
        mu3 = mu2 - 5 * np.random.rand(dim)
        cov1 = 1.5 * np.eye(dim)
        cov2 = 2 * np.eye(dim)
        cov3 = 3 * np.eye(dim)

        coef_xz = np.reshape(2 * np.random.rand(dim) - 1, (dim, 1))
        coef_u1x = np.reshape(np.random.rand(dim) + 3, (dim, 1))
        coef_u3x = np.reshape(-2 * np.random.rand(dim) - 2, (dim, 1))
        coef_zy = np.reshape(10 * np.random.rand(dim) - 5, (dim, 1))

        U1 = np.random.multivariate_normal(mu1, cov1, N)
        U2 = np.random.multivariate_normal(mu2, cov2, N)
        U3 = np.random.multivariate_normal(mu3, cov3, N)

        Z = U1 + U2
        X = intfy(
            np.round(inverse_logit(np.matrix(Z) * coef_xz + np.matrix(U1) * coef_u1x + np.matrix(U3) * coef_u3x - 1),
                     0))
        Y = np.array(np.matrix(Z) * coef_zy) + \
            np.reshape(np.array(U2[:, dim - 1]), (N, 1)) + \
            np.reshape(np.array(U3[:, 1]), (N, 1)) + \
            100 * np.reshape(X, (N, 1))

        X_intv = intfy(np.asarray([0] * int(N / 2) + [1] * int(N / 2)))
        Y_intv = np.array(np.matrix(Z) * coef_zy) + \
                 np.reshape(np.array(U2[:, dim - 1]), (N, 1)) + \
                 np.reshape(np.array(U3[:, 1]), (N, 1)) + \
                 100 * np.reshape(X_intv, (N, 1))

        X_obs = np.asarray(X)
        Y_obs = np.asarray(Y)
        Z_obs = np.asarray(Z)

        Obs_X = pd.DataFrame(X_obs)
        Obs_Y = pd.DataFrame(Y_obs)
        Obs_Z = pd.DataFrame(Z_obs)

        Obs_XY = pd.concat([Obs_X, Obs_Y], axis=1)
        Obs_XY.columns = ['X', 'Y']
        Obs_Z.columns = range(dim)
        Obs = pd.concat([Obs_XY, Obs_Z], axis=1)

        X_intv = np.asarray(X_intv);
        Y_intv = np.asarray(Y_intv)
        Intv_X = pd.DataFrame(X_intv)
        Intv_Y = pd.DataFrame(Y_intv)
        Intv_Z = pd.DataFrame(Z_obs)

        Intv_XY = pd.concat([Intv_X, Intv_Y], axis=1)
        Intv_XY.columns = ['X', 'Y']
        Intv_Z.columns = range(dim)
        Intv = pd.concat([Intv_XY, Intv_Z], axis=1)

        sample_indces_x1 = np.random.choice(list(range(0, int(N / 2))), int(Ns / 2), replace=False)
        sample_indces_x0 = np.random.choice(list(range(int(N / 2), N)), int(Ns / 2), replace=False)
        sample_indices = np.asarray(list(sample_indces_x1) + list(sample_indces_x0))

        X_sintv = X_intv[sample_indices]
        Y_sintv = Y_intv[sample_indices];
        Z_sintv = Z_obs[sample_indices]
        SIntv_X = pd.DataFrame(X_sintv)
        SIntv_Y = pd.DataFrame(Y_sintv)
        SIntv_Z = pd.DataFrame(Z_sintv)

        SIntv_XY = pd.concat([SIntv_X, SIntv_Y], axis=1)
        SIntv_XY.columns = ['X', 'Y']
        SIntv_Z.columns = range(dim)
        Intv_S = pd.concat([SIntv_XY, SIntv_Z], axis=1)

        return Obs, Intv, Intv_S

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
T = 10000 # number of rounds
N = 10000 # Number of data to be sampled (for accurate interventional effect)
Ns = 50 # Number of Ds
dim = 2 # Dimension. Only matter for case_num = 4
case_num = 31
Obs, Intv, Intv_S =  Data_generation(case_num=case_num ,N=N,Ns=Ns,dim = dim,seed_num=123)

true_coef = np.matrix( linear_training(Obs[['X','Z']],Obs['Y']).coef_ )


# If not case 1, (Y continuous)
if case_num != 1:
    std_sydox0 = np.std(Intv_S[Intv_S['X'] == 0]['Y'])
    std_sydox1 = np.std(Intv_S[Intv_S['X'] == 1]['Y'])
    std_sydox = [std_sydox0, std_sydox1]

    std_yx0 = np.std(Obs[Obs['X']==0]['Y'])
    std_yx1 = np.std(Obs[Obs['X'] == 1]['Y'])
    std_yx = [std_yx0, std_yx1]

# # Compute LB, UB using causal bound technique.
# ## Not necessary for KB-UCB.
# LB0, p_true0, UB0 = Causal_bound(case_num=case_num , dim=dim, Obs=Obs, Intv=Intv, Intv_S = Intv_S, x=0, z=0)
# LB1, p_true1, UB1 = Causal_bound(case_num=case_num , dim=dim, Obs=Obs, Intv=Intv, Intv_S = Intv_S, x=1, z=0)
# LB = [LB0,LB1]
# L_max = np.max(LB)
# UB = [UB0,UB1]

K = 2 # Number of arms.
Arm = []
Reward = []

# Number of selection of arm a at time t
## if we choose a at time t, then Na_T +=1 from t+1
Na_T = dict()
sum_reward = 0 # sum of rewards by far to compute cum.regret.
delta = 0.01

alpha = np.sqrt(  0.5 * np.log( 2*T*K/delta ) )


''' Bandit start! '''
S = int( round(np.log(T)) + 1 )
Psi = dict()
for s_idx in range(1,S):
    Psi[s_idx] = [0]

Bandit = {'Arm':[], 'Reward':[], 'Cum_regret':[],'Prob_opt':[]}
Na_T = dict()
for a in range(K):
    Na_T[a] = 0

sum_regret = 0
for t in range(1,T+1):
    s = 1
    M = range(K)
    at = -1
    width = []
    rewards = []
    UBs = []
    z = Obs['Z'].loc[t]
    true_rewards = []
    for a in range(K):
        x_ta = np.matrix([a,z])
        true_rewards.append( np.array( x_ta * true_coef.T )[0][0] )
    r_star = max(true_rewards)
    opt_arm = np.argmax(true_rewards)

    while(1):
        width, rewards, UBs = Baseline_UCB(case_num,alpha,np.unique(Psi[s]),Obs,dim,K,t)

        if all(width[idx] < 1/np.sqrt(T) for idx in M):
            at = np.argmax(UBs)
            rt = max(rewards)
            ub = max(UBs)
            for s_idx in range(1,S):
                Psi[s_idx].append( Psi[s_idx][t-1] )
            break
        elif all(width[idx] < (2**(-s)) for idx in M):
            M2 = []
            for a in M:
                if UBs[a] > max(UBs) - 2**(1-s):
                    M2.append(a)
            s += 1
            M = M2
        else:
            gen = [x for x in M if width[x] > 2**(-s)  ]
            at = np.argmax( np.array(UBs)[gen] )
            UB = UBs[at]
            rt = rewards[at]

            for s_idx in range(1,S):
                if s_idx == s:
                    Psi[s_idx].append(t)
                else:
                    Psi[s_idx].append( Psi[s_idx][t-1] )

        if at != -1:
            break

    Na_T[at] += 1
    sum_regret += r_star - rt
    Bandit['Arm'].append(at)
    Bandit['Reward'].append(rt)
    Bandit['Cum_regret'].append(sum_regret)
    Bandit['Prob_opt'].append( Na_T[opt_arm]/t )

    print(t, at, round(rt, 3), round(sum_regret), round(Na_T[opt_arm]/t, 3))


Bandit = pd.DataFrame(Bandit)

# Graphical illustration
f, ax = plt.subplots(2, sharex='all')
plt.title("KL-UCB")
ax[0].set_title('Cumulative regret')
ax[0].plot(Bandit['Cum_regret'])

ax[1].set_title('Probability of choosing opt arm')
ax[1].plot(Bandit['Prob_opt'])




