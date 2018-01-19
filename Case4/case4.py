import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.preprocessing import normalize

def inverse_logit(Z):
    return np.exp(Z) / (np.exp(Z)+1)

def intfy(W):
    return np.array(list(map(int,W)))

print("----------------------START------------------------------")

''' Data generation '''
''' Data generation '''
np.random.seed(123)

fig_version = 2

dim = 100
# Ns = 10*dim
Ns = 200
N = 100*dim

mu1 = np.random.rand(dim)
mu2 = np.random.rand(dim)
mu3 = np.random.rand(dim)
cov1 = np.dot(np.random.rand(dim, dim), np.random.rand(dim, dim).transpose())
cov2 = np.dot(np.random.rand(dim, dim), np.random.rand(dim, dim).transpose())
cov3 = np.dot(np.random.rand(dim, dim), np.random.rand(dim, dim).transpose())

coef_xz = np.reshape(1 * np.random.rand(dim), (dim, 1))
coef_u1x = np.reshape(1 * np.random.rand(dim), (dim, 1))
coef_u3x = np.reshape(1 * np.random.rand(dim), (dim, 1))
coef_zy = np.reshape(1 * np.random.rand(dim), (dim, 1))
coef_u2y = np.reshape(1 * np.random.rand(dim), (dim, 1))
coef_u3y = np.reshape(1 * np.random.rand(dim), (dim, 1))

U1 = np.random.multivariate_normal(mu1, cov1, N)
U2 = np.random.multivariate_normal(mu2, cov2, N)
U3 = np.random.multivariate_normal(mu3, cov3, N)

Z = normalize(1 * U1 + 1 * U2)
sample_indices_disc = np.random.choice(list(range(0, dim)), int(dim / 3), replace=False)
Z_replace = np.random.binomial(1, 0.5, (N, len(sample_indices_disc)))
Z[:, sample_indices_disc] = Z_replace

X = intfy(np.round(
    inverse_logit(normalize(np.matrix(Z) * coef_xz + np.matrix(U1) * coef_u1x + np.matrix(U3) * coef_u3x)), 0))

Y_orig = np.array(np.matrix(Z) * coef_zy) + \
         np.array(np.matrix(U2) * coef_u2y) + \
         np.array(np.matrix(U3) * coef_u3y)

Y_norm = (Y_orig - np.mean(Y_orig)) / np.std(Y_orig)
Y = Y_norm + np.reshape(0.5 * X, (N, 1))
# Y = np.round( inverse_logit(Y), 0)

X_intv = intfy(np.asarray([0] * int(N / 2) + [1] * int(N / 2)))
Y_intv = Y_norm + np.reshape(0.5 * X_intv, (N, 1))

# coef_xz = np.reshape(-1 * np.random.rand(dim), (dim, 1))
# coef_u1x = np.reshape(-1 * np.random.rand(dim), (dim, 1))
# coef_u3x = np.reshape(1 * np.random.rand(dim), (dim, 1))
# coef_zy = np.reshape(2 * np.random.rand(dim), (dim, 1))
#
# U1 = np.random.multivariate_normal(mu1, cov1, N)
# U2 = np.random.multivariate_normal(mu2, cov2, N)
# U3 = np.random.multivariate_normal(mu3, cov3, N)
#
# Z = normalize(2*U1 + 1*U2)
# sample_indices_disc = np.random.choice(list(range(0, dim)), int(dim / 3), replace=False)
# Z_replace = np.random.binomial(1, 0.5, (N, len(sample_indices_disc)))
# Z[:, sample_indices_disc] = Z_replace
#
# X = intfy(np.round(
#     inverse_logit(normalize(np.matrix(Z) * coef_xz + np.matrix(U1) * coef_u1x + np.matrix(U3) * coef_u3x)), 0))
# Y = np.array(np.matrix(Z) * coef_zy) + \
#     np.reshape(np.array(U2[:, dim - 1]), (N, 1)) + \
#     np.reshape(np.array(U3[:, 1]), (N, 1)) + \
#     15 * np.reshape(X, (N, 1))
#
# X_intv = intfy(np.asarray([0] * int(N / 2) + [1] * int(N / 2)))
# Y_intv = np.array(np.matrix(Z) * coef_zy) + \
#          np.reshape(np.array(U2[:, dim - 1]), (N, 1)) + \
#          np.reshape(np.array(U3[:, 1]), (N, 1)) + \
#          15 * np.reshape(X_intv, (N, 1))

X_obs = np.asarray(X)
Y_obs = np.asarray(Y)
Z_obs = np.asarray(Z)

Obs_X = pd.DataFrame(X_obs)
Obs_Y = pd.DataFrame(Y_obs)
Obs_Z = pd.DataFrame(Z_obs)

Obs_XY = pd.concat([Obs_X,Obs_Y],axis=1)
Obs_XY.columns = ['X','Y']
Obs_Z.columns = range(dim)
Obs = pd.concat([Obs_XY,Obs_Z],axis=1)

X_intv = np.asarray(X_intv); Y_intv = np.asarray(Y_intv)
Intv_X = pd.DataFrame(X_intv)
Intv_Y = pd.DataFrame(Y_intv)
Intv_Z = pd.DataFrame(Z_obs)

Intv_XY = pd.concat([Intv_X,Intv_Y],axis=1)
Intv_XY.columns = ['X','Y']
Intv_Z.columns = range(dim)
Intv = pd.concat([Intv_XY,Intv_Z],axis=1)

sample_indces_x1 = np.random.choice(list(range(0,int(N/2))),int(Ns/2),replace=False)
sample_indces_x0 = np.random.choice(list(range(int(N/2),N)),int(Ns/2),replace=False)
sample_indices = np.asarray(list(sample_indces_x1) + list(sample_indces_x0))

X_sintv = X_intv[sample_indices]
Y_sintv = Y_intv[sample_indices]; Z_sintv = Z_obs[sample_indices]
SIntv_X = pd.DataFrame(X_sintv)
SIntv_Y = pd.DataFrame(Y_sintv)
SIntv_Z = pd.DataFrame(Z_sintv)

SIntv_XY = pd.concat([SIntv_X,SIntv_Y],axis=1)
SIntv_XY.columns = ['X','Y']
SIntv_Z.columns = range(dim)
Intv_S = pd.concat([SIntv_XY,SIntv_Z],axis=1)


if fig_version == 1:
    # Fix X=x
    x_care = 1

    # Compute mu, std of P(Y=y | x)
    Obs_yx = Obs[Obs['X'] == x_care]['Y']
    mu_yx = np.mean(Obs_yx)
    std_yx = np.std(Obs_yx)

    # Compute mu of P(Y=y | do(x))
    Intv_yx = Intv[Intv['X'] == x_care]['Y']
    mu_ydox = np.mean(Intv_yx)
    std_ydox = np.std(Intv_yx)

    # Compute std of Ds(Y|do(x))
    Intv_S_yx = Intv_S[Intv_S['X'] == x_care]['Y']
    std_sydox = np.std(Intv_S_yx)

    # Compute P(X=x_care)
    px = len(Obs[Obs['X'] == x_care]) / N
        ## Compute -log(P(x))
    Hx = -np.log(px)
    C = Hx + 0.5 - np.log(std_sydox / std_yx) - (std_yx ** 2) / (2 * std_sydox ** 2)

    # Compute LB and UB
    LB = mu_yx - std_sydox * np.sqrt(2 * C)
    UB = mu_yx + std_sydox * np.sqrt(2 * C)

    print("----------- Analysis --------")
    print("P(x): ", px)
    print("min(P(Y|x)): ", min(norm.pdf(Obs_yx, loc=mu_yx, scale=std_yx)))
    print("avg(P(Y|x)): ", np.mean(norm.pdf(Obs_yx, loc=mu_yx, scale=std_yx)))
    print("min(P(Y,x)): ", px * min(norm.pdf(Obs_yx, loc=mu_yx, scale=std_yx)))
    print("avg(P(Y,x)): ", px * np.mean(norm.pdf(Obs_yx, loc=mu_yx, scale=std_yx)))

    print("X=", x_care, ", Interval:", LB, mu_ydox, UB)

    # Graphical illustration
    domain = np.linspace(mu_yx - (3*C) * std_yx, mu_yx + (3*C) * std_yx, num=10000)
    output = ((mu_yx - domain) ** 2) / (2 * (std_sydox ** 2))
    plt.figure(1)
    plt.plot(domain, output)
    plt.plot(domain, [C] * len(domain))
    plt.axvline(mu_ydox)

elif fig_version == 2:

    ''' When X = 0 '''
    # Fix X=x
    x_care = 0

    # Compute mu, std of P(Y=y | x)
    Obs_yx0 = Obs[Obs['X'] == x_care]['Y']
    mu_yx0 = np.mean(Obs_yx0)
    std_yx0 = np.std(Obs_yx0)

    # Compute mu of P(Y=y | do(x))
    Intv_yx0 = Intv[Intv['X'] == x_care]['Y']
    mu_ydox0 = np.mean(Intv_yx0)

    # Compute std of Ds(Y|do(x))
    Intv_S_yx0 = Intv_S[Intv_S['X'] == x_care]['Y']
    std_sydox0 = np.std(Intv_S_yx0)

    # Compute P(X=x_care)
    px0 = len(Obs[Obs['X'] == x_care]) / N
    ## Compute -log(P(x))
    Hx0 = -np.log(px0)
    C0 = Hx0 + 0.5 - np.log(std_sydox0 / std_yx0) - (std_yx0 ** 2) / (2 * std_sydox0 ** 2)

    # Compute LB and UB
    LB0 = mu_yx0 - std_sydox0 * np.sqrt(2 * C0)
    UB0 = mu_yx0 + std_sydox0 * np.sqrt(2 * C0)

    print("----------- Analysis of X = 0 --------")
    print("P(x): ", px0)
    print("min(P(Y|x)): ", min(norm.pdf(Obs_yx0, loc=mu_yx0, scale=std_yx0)))
    print("avg(P(Y|x)): ", np.mean(norm.pdf(Obs_yx0, loc=mu_yx0, scale=std_yx0)))
    print("min(P(Y,x)): ", px0 * min(norm.pdf(Obs_yx0, loc=mu_yx0, scale=std_yx0)))
    print("avg(P(Y,x)): ", px0 * np.mean(norm.pdf(Obs_yx0, loc=mu_yx0, scale=std_yx0)))

    print("Interval when X=0:", LB0, mu_ydox0, UB0)



    ''' When X = 1 '''
    x_care = 1

    # Compute mu, std of P(Y=y | x)
    Obs_yx1 = Obs[Obs['X'] == x_care]['Y']
    mu_yx1 = np.mean(Obs_yx1)
    std_yx1 = np.std(Obs_yx1)

    # Compute mu of P(Y=y | do(x))
    Intv_yx1 = Intv[Intv['X'] == x_care]['Y']
    mu_ydox1 = np.mean(Intv_yx1)

    # Compute std of Ds(Y|do(x))
    Intv_S_yx1 = Intv_S[Intv_S['X'] == x_care]['Y']
    std_sydox1 = np.std(Intv_S_yx1)

    # Compute P(X=x_care)
    px1 = len(Obs[Obs['X'] == x_care]) / N
    ## Compute -log(P(x))
    Hx1 = -np.log(px1)
    C1 = Hx1 + 0.5 - np.log(std_sydox1 / std_yx1) - (std_yx1 ** 2) / (2 * std_sydox1 ** 2)

    # Compute LB and UB
    LB1 = mu_yx1 - std_sydox1 * np.sqrt(2 * C1)
    UB1 = mu_yx1 + std_sydox1 * np.sqrt(2 * C1)

    print("----------- Analysis of X = 1 --------")
    print("P(x): ", px1)
    print("min(P(Y|x)): ", min(norm.pdf(Obs_yx1, loc=mu_yx1, scale=std_yx1)))
    print("avg(P(Y|x)): ", np.mean(norm.pdf(Obs_yx1, loc=mu_yx1, scale=std_yx1)))
    print("min(P(Y,x)): ", px1 * min(norm.pdf(Obs_yx1, loc=mu_yx1, scale=std_yx1)))
    print("avg(P(Y,x)): ", px1 * np.mean(norm.pdf(Obs_yx1, loc=mu_yx1, scale=std_yx1)))

    print("Interval when X=1:", LB1, mu_ydox1, UB1)

    # Graphical illustration
    f, ax = plt.subplots(2, sharex='all')
    min_val = min(LB0, LB1, min(Obs['Y']))
    max_val = max(UB0, UB1, max(Obs['Y']))

    x_domain = np.linspace(min_val, max_val, 10000)
    y_graphic = 0.5
    y_domain = np.array([y_graphic] * len(x_domain))

    title_size = 40
    label_size = 30
    legend_size = 25
    marker_size = 15

    ax[0].set_title('Interval of P(y|do(X= 0))', fontsize=title_size)
    ax[0].plot(x_domain, y_domain)
    ax[0].axvline(x=min_val, ymin=0.48, ymax=0.52)
    ax[0].axvline(x=max_val, ymin=0.48, ymax=0.52)
    ax[0].axvline(x=LB0, ymin=0.35, ymax=0.65, label="Lower bound")
    ax[0].axvline(x=UB0, ymin=0.35, ymax=0.65, label="Upper bound")
    ax[0].plot(mu_ydox0, y_graphic, marker='o', color='r', label="P(y|do(X=0))",markersize=marker_size)
    ax[0].set_xlabel('Mean of causal quantity', fontsize=label_size)  # X label
    ax[0].yaxis.set_visible(False)  # Hide only x axis
    ax[0].xaxis.set_visible(False)  # Hide only x axis
    ax[0].legend(fontsize=legend_size)

    ax[1].plot(x_domain, y_domain)
    ax[1].set_title('Interval of P(y|do(X= 1))', fontsize=title_size)
    ax[1].axvline(x=min_val, ymin=0.48, ymax=0.52)
    ax[1].axvline(x=max_val, ymin=0.48, ymax=0.52)
    ax[1].axvline(x=LB1, ymin=0.35, ymax=0.65, label="Lower bound")
    ax[1].axvline(x=UB1, ymin=0.35, ymax=0.65, label="Upper bound")
    ax[1].plot(mu_ydox1, y_graphic, marker='o', color='r', label="P(y|do(X=1))",markersize=marker_size)
    ax[1].set_xlabel('Mean of causal quantity', fontsize=label_size)  # X label
    ax[1].yaxis.set_visible(False)  # Hide only x axis
    ax[1].legend(fontsize=legend_size)