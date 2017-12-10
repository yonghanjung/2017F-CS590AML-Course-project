import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm

def inverse_logit(Z):
    return np.exp(Z) / (np.exp(Z)+1)

def intfy(W):
    return np.array(list(map(int,W)))


print("----------------------START------------------------------")

''' Data generation '''
np.random.seed(1)
N = 1000; Ns = 50
fig_version = 2

U1 = np.random.normal(-20,20,N)
U2 = np.random.normal(50,20,N)
U3 = np.random.normal(40,20,N)

Z = U1 + U2
X = intfy( np.round( inverse_logit(2*Z + 4*U1 + 4*U3 - 1), 0) )
Y = Z + U2 + U3 + 100*X

X_intv = intfy( np.asarray([0] * int(N/2) + [1] * int(N/2)) )
Y_intv = Z + U2 + U3 + 100*X_intv

X_obs = np.asarray(X)
Y_obs = np.asarray(Y)
Z_obs = np.asarray(Z)
Obs = pd.DataFrame({'Z':Z_obs,'X':X_obs,'Y':Y_obs})

X_intv = np.asarray(X_intv); Y_intv = np.asarray(Y_intv)
Intv = pd.DataFrame({'X':X_intv,'Y':Y_intv, 'Z':Z_obs})

sample_indces_x1 = np.random.choice(list(range(0,int(N/2))),int(Ns/2),replace=False)
sample_indces_x0 = np.random.choice(list(range(int(N/2),N)),int(Ns/2),replace=False)
sample_indices = np.asarray(list(sample_indces_x1) + list(sample_indces_x0))

X_sintv = X_intv[sample_indices]
Y_sintv = Y_intv[sample_indices]; Z_sintv = Z_obs[sample_indices]
Intv_S = pd.DataFrame({'X':X_sintv, 'Y':Y_sintv, 'Z':Z_sintv})


if fig_version == 1:

    # Fix X=x
    x_care = 0

    # Compute mu, std of P(Y=y | x)
    Obs_yx = Obs[Obs['X']==x_care]['Y']
    mu_yx = np.mean(Obs_yx)
    std_yx = np.std(Obs_yx)

    # Compute mu of P(Y=y | do(x))
    Intv_yx = Intv[Intv['X'] == x_care]['Y']
    mu_ydox = np.mean(Intv_yx)

    # Compute std of Ds(Y|do(x))
    Intv_S_yx = Intv_S[Intv_S['X']==x_care]['Y']
    std_sydox = np.std(Intv_S_yx)

    # Compute P(X=x_care)
    px = len(Obs[Obs['X']==x_care])/N
    ## Compute -log(P(x))
    Hx = -np.log(px)
    C = Hx + 0.5 - np.log(std_sydox / std_yx) - (std_yx**2)/(2*std_sydox** 2)

    # Compute LB and UB
    LB = mu_yx - std_sydox*np.sqrt(2*C)
    UB = mu_yx + std_sydox*np.sqrt(2*C)

    print("----------- Analysis --------")
    print("P(x): ", px)
    print("min(P(Y|x)): ", min(norm.pdf(Obs_yx,loc=mu_yx,scale=std_yx)) )
    print("avg(P(Y|x)): ", np.mean(norm.pdf(Obs_yx, loc=mu_yx, scale=std_yx)))
    print("min(P(Y,x)): ", px * min(norm.pdf(Obs_yx, loc=mu_yx, scale=std_yx)))
    print("avg(P(Y,x)): ", px * np.mean(norm.pdf(Obs_yx, loc=mu_yx, scale=std_yx)))

    # print("P(y|x): ", pz)
    # print("P(x,z): ", pxz)
    # print("P(x,y,z) for y=1: ", px_y1_z)
    # print("P(x,y,z) for y=0: ", px_y0_z)

    print("X=", x_care, ", Interval:", LB, mu_ydox, UB)

    # Graphical illustration
    domain = np.linspace(mu_yx - 3*C*std_yx, mu_yx + 3*C*std_yx, num=10000)
    output = ((mu_yx-domain)**2) /(2*(std_sydox**2))
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
    # Fix X=x
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
    C1 = Hx1 + 1.5 - np.log(std_sydox1 / std_yx1) - (std_yx1 ** 2) / (2 * std_sydox1 ** 2)

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



    ax[0].set_title('Interval of P(y|do(X= 0))')
    ax[0].plot(x_domain, y_domain)
    ax[0].axvline(x=min_val, ymin=0.48, ymax=0.52)
    ax[0].axvline(x=max_val, ymin=0.48, ymax=0.52)
    ax[0].axvline(x=LB0, ymin=0.35, ymax=0.65)
    ax[0].axvline(x=UB0, ymin=0.35, ymax=0.65)
    ax[0].plot(mu_ydox0, y_graphic, marker='o', color='r')

    ax[1].plot(x_domain, y_domain)
    ax[1].set_title('Interval of P(y|do(X= 1))')
    ax[1].axvline(x=min_val, ymin=0.48, ymax=0.52)
    ax[1].axvline(x=max_val, ymin=0.48, ymax=0.52)
    ax[1].axvline(x=LB1, ymin=0.35, ymax=0.65)
    ax[1].axvline(x=UB1, ymin=0.35, ymax=0.65)
    ax[1].plot(mu_ydox1, y_graphic, marker='o', color='r')
