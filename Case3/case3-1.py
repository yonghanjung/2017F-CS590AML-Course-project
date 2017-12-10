import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn import linear_model
from scipy.stats import norm

def logistic_training(X,Y):
    clf = linear_model.LogisticRegression()
    clf.fit(X,Y)
    return clf

def linear_training(X,Y):
    clf = linear_model.LinearRegression()
    clf.fit(X, Y)
    return clf

def inverse_logit(Z):
    return np.exp(Z) / (np.exp(Z)+1)

def causal_bound(p_obs,Hx,minmax_mode):
    def fun(x):
        if minmax_mode == 'min':
            return x
        elif minmax_mode == 'max':
            return -x


    cons = ({'type': 'ineq',
             'fun': lambda x: -( p_obs * np.log(p_obs / x) + (1 - p_obs) * np.log((1 - p_obs) / (1 - x)) - Hx )},
            {'type': 'ineq',
             'fun': lambda x: (p_obs * np.log(p_obs / x) + (1 - p_obs) * np.log((1 - p_obs) / (1 - x)))}
            )

    lbounds = 0.00001
    ubounds = 0.99999
    bnds = [(lbounds, ubounds)]

    if minmax_mode == 'min':
        x0 = 0.01
    elif minmax_mode == 'max':
        x0 = 0.99

    res = minimize(fun, x0=x0, constraints=cons, method='SLSQP',
                       bounds=bnds)

    if minmax_mode == 'min':
        return round(res.fun,3)
    elif minmax_mode == 'max':
        return round(-res.fun,3)

def intfy(W):
    return np.array(list(map(int,W)))

print("----------------------START------------------------------")

''' Data generation '''
np.random.seed(1)
N = 100000; Ns = 100
fig_version = 2

U1 = np.random.normal(5,10,N)
U2 = np.random.normal(-20,10,N)
U3 = np.random.normal(10,30,N)

Z = U1 + U2
X = intfy( np.round( inverse_logit(2*Z + 3*U1 + 4*U3 - 1), 0) )
Y = Z + U2 + U3 + 10*X

X_intv = intfy( np.asarray([0] * int(N/2) + [1] * int(N/2)) )
Y_intv = Z + U2 + U3 + 10*X_intv

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

z_care = -10

if fig_version == 1:
    x_care = 1
    care = [x_care, z_care]

    # Compute mu and std of P(Y|x,z)
    ## Training for P(Y|x,z) for all x and z
    obs_lin = linear_training(Obs[['X','Z']],Obs['Y'])
    ### Compute the residual distribution for std
    obs_result = obs_lin.predict(Obs[['X', 'Z']])
    obs_resid = obs_lin.predict(Obs[['X','Z']]) - Obs['Y']
    ## Compute mu and std of P(Y|x,z)
    mu_yxz = obs_lin.predict(care)[0]
    std_yxz = np.std(obs_resid)

    # Compute mu and std of Ds(Y|do(x),z)
    ## Training for D(Y|do(x),z) for all x and z
    Sdox_lin = linear_training(Intv_S[['X','Z']],Intv_S['Y'])
    ### Compute the residual distribution for std
    Sdox_resid = Sdox_lin.predict(Intv_S[['X', 'Z']]) - Intv_S['Y']
    ## Compute mu and std of Ds(Y|do(x),z)
    mu_sydox = Sdox_lin.predict(care)[0]
    std_sydox = np.std(Sdox_resid)

    # Compute mu and std of P(Y|do(x),z)
    ## Training for P(Y|do(x),z) for all x and z
    dox_lin = linear_training(Intv[['X', 'Z']], Intv['Y'])
    ### Compute the residual distribution for std
    dox_resid = dox_lin.predict(Intv[['X', 'Z']]) - Intv['Y']
    ## Compute mu and std of P(Y|do(x),z)
    mu_ydox = dox_lin.predict(care)[0]
    std_ydox = np.std(Sdox_resid)

    # Compute P(X=x_care | z)
    ## Training P(X|Z) for all z and x
    pxz_logit = logistic_training(Obs[['Z']],Obs['X'])
    ## Compute P(X|z)
    pxz_compute = pxz_logit.predict_proba(z_care)[0]
    ## Pick P(x_care | z)
    pxz = pxz_compute[x_care]

    # Compute constant term C = Hx + 1/2 - log(std_yxz / std_ydoxz) - (std_yxz)^2/(2 * (std_ydoxz)^2)
    Hxz = -np.log(pxz)
    C = Hxz + 0.5 - np.log(std_sydox / std_yxz) - (std_yxz ** 2) / (2 * std_sydox ** 2)

    # Compute LB and UB
    LB = mu_yxz - std_sydox * np.sqrt(2 * C)
    UB = mu_yxz + std_sydox * np.sqrt(2 * C)

    print("----------- Analysis --------")
    print("P(x): ", pxz)
    print("min(P(Y|x)): ", min(norm.pdf(obs_result, loc=mu_yxz, scale=std_yxz)))
    print("avg(P(Y|x)): ", np.mean(norm.pdf(obs_result, loc=mu_yxz, scale=std_yxz)))
    print("min(P(Y,x)): ", pxz * min(norm.pdf(obs_result, loc=mu_yxz, scale=std_yxz)))
    print("avg(P(Y,x)): ", pxz * np.mean(norm.pdf(obs_result, loc=mu_yxz, scale=std_yxz)))

    print("X=", x_care, ", Interval:", LB, mu_ydox, UB)

    # Graphical illustration
    domain = np.linspace(mu_yxz - 10 * std_yxz, mu_yxz + 10 * std_yxz, num=10000)
    output = ((mu_yxz - domain) ** 2) / (2 * (std_sydox ** 2))
    plt.figure(1)
    plt.plot(domain, output)
    plt.plot(domain, [C] * len(domain))
    plt.axvline(mu_ydox)

if fig_version == 2:
    ''' When X = 0 '''
    x_care = 0
    care = [x_care, z_care]

    # Compute mu and std of P(Y|x,z)
    ## Training for P(Y|x,z) for all x and z
    obs_lin0 = linear_training(Obs[['X', 'Z']], Obs['Y'])
    ### Compute the residual distribution for std
    obs_result0 = obs_lin0.predict(Obs[['X', 'Z']])
    obs_resid0 = obs_lin0.predict(Obs[['X', 'Z']]) - Obs['Y']
    ## Compute mu and std of P(Y|x,z)
    mu_yxz0 = obs_lin0.predict(care)[0]
    std_yxz0 = np.std(obs_resid0)

    # Compute mu and std of Ds(Y|do(x),z)
    ## Training for D(Y|do(x),z) for all x and z
    Sdox_lin0 = linear_training(Intv_S[['X', 'Z']], Intv_S['Y'])
    ### Compute the residual distribution for std
    Sdox_resid0 = Sdox_lin0.predict(Intv_S[['X', 'Z']]) - Intv_S['Y']
    ## Compute mu and std of Ds(Y|do(x),z)
    mu_sydox0 = Sdox_lin0.predict(care)[0]
    std_sydox0 = np.std(Sdox_resid0)

    # Compute mu and std of P(Y|do(x),z)
    ## Training for P(Y|do(x),z) for all x and z
    dox_lin0 = linear_training(Intv[['X', 'Z']], Intv['Y'])
    ### Compute the residual distribution for std
    dox_resid0 = dox_lin0.predict(Intv[['X', 'Z']]) - Intv['Y']
    ## Compute mu and std of P(Y|do(x),z)
    mu_ydox0 = dox_lin0.predict(care)[0]
    std_ydox0 = np.std(Sdox_resid0)

    # Compute P(X=x_care | z)
    ## Training P(X|Z) for all z and x
    pxz_logit0 = logistic_training(Obs[['Z']], Obs['X'])
    ## Compute P(X|z)
    pxz_compute0 = pxz_logit0.predict_proba(z_care)[0]
    ## Pick P(x_care | z)
    pxz0 = pxz_compute0[x_care]

    # Compute constant term C = Hx + 1/2 - log(std_yxz / std_ydoxz) - (std_yxz)^2/(2 * (std_ydoxz)^2)
    Hxz0 = -np.log(pxz0)
    C0 = Hxz0 + 0.5 - np.log(std_sydox0 / std_yxz0) - (std_yxz0 ** 2) / (2 * std_sydox0 ** 2)

    # Compute LB and UB
    LB0 = mu_yxz0 - std_sydox0 * np.sqrt(2 * C0)
    UB0 = mu_yxz0 + std_sydox0 * np.sqrt(2 * C0)

    print("----------- Analysis --------")
    print("P(x): ", pxz0)
    print("min(P(Y|x)): ", min(norm.pdf(obs_result0, loc=mu_yxz0, scale=std_yxz0)))
    print("avg(P(Y|x)): ", np.mean(norm.pdf(obs_result0, loc=mu_yxz0, scale=std_yxz0)))
    print("min(P(Y,x)): ", pxz0 * min(norm.pdf(obs_result0, loc=mu_yxz0, scale=std_yxz0)))
    print("avg(P(Y,x)): ", pxz0 * np.mean(norm.pdf(obs_result0, loc=mu_yxz0, scale=std_yxz0)))

    print("X=", x_care, ", Interval:", LB0, mu_ydox0, UB0)




    ''' When X = 1 '''
    x_care = 1
    care = [x_care, z_care]

    # Compute mu and std of P(Y|x,z)
    ## Training for P(Y|x,z) for all x and z
    obs_lin1= linear_training(Obs[['X', 'Z']], Obs['Y'])
    ### Compute the residual distribution for std
    obs_result1 = obs_lin1.predict(Obs[['X', 'Z']])
    obs_resid1 = obs_lin1.predict(Obs[['X', 'Z']]) - Obs['Y']
    ## Compute mu and std of P(Y|x,z)
    mu_yxz1 = obs_lin1.predict(care)[0]
    std_yxz1 = np.std(obs_resid1)

    # Compute mu and std of Ds(Y|do(x),z)
    ## Training for D(Y|do(x),z) for all x and z
    Sdox_lin1 = linear_training(Intv_S[['X', 'Z']], Intv_S['Y'])
    ### Compute the residual distribution for std
    Sdox_resid1 = Sdox_lin1.predict(Intv_S[['X', 'Z']]) - Intv_S['Y']
    ## Compute mu and std of Ds(Y|do(x),z)
    mu_sydox1 = Sdox_lin1.predict(care)[0]
    std_sydox1 = np.std(Sdox_resid1)

    # Compute mu and std of P(Y|do(x),z)
    ## Training for P(Y|do(x),z) for all x and z
    dox_lin1 = linear_training(Intv[['X', 'Z']], Intv['Y'])
    ### Compute the residual distribution for std
    dox_resid1 = dox_lin1.predict(Intv[['X', 'Z']]) - Intv['Y']
    ## Compute mu and std of P(Y|do(x),z)
    mu_ydox1 = dox_lin1.predict(care)[0]
    std_ydox1 = np.std(Sdox_resid1)

    # Compute P(X=x_care | z)
    ## Training P(X|Z) for all z and x
    pxz_logit1 = logistic_training(Obs[['Z']], Obs['X'])
    ## Compute P(X|z)
    pxz_compute1 = pxz_logit1.predict_proba(z_care)[0]
    ## Pick P(x_care | z)
    pxz1 = pxz_compute1[x_care]

    # Compute constant term C = Hx + 1/2 - log(std_yxz / std_ydoxz) - (std_yxz)^2/(2 * (std_ydoxz)^2)
    Hxz1 = -np.log(pxz1)
    C1 = Hxz1 + 0.5 - np.log(std_sydox1 / std_yxz1) - (std_yxz1 ** 2) / (2 * std_sydox1 ** 2)

    # Compute LB and UB
    LB1 = mu_yxz1 - std_sydox1 * np.sqrt(2 * C1)
    UB1 = mu_yxz1 + std_sydox1 * np.sqrt(2 * C1)

    print("----------- Analysis --------")
    print("P(x): ", pxz1)
    print("min(P(Y|x)): ", min(norm.pdf(obs_result1, loc=mu_yxz1, scale=std_yxz1)))
    print("avg(P(Y|x)): ", np.mean(norm.pdf(obs_result1, loc=mu_yxz1, scale=std_yxz1)))
    print("min(P(Y,x)): ", pxz1 * min(norm.pdf(obs_result1, loc=mu_yxz1, scale=std_yxz1)))
    print("avg(P(Y,x)): ", pxz1 * np.mean(norm.pdf(obs_result1, loc=mu_yxz1, scale=std_yxz1)))

    print("X=", x_care, ", Interval:", LB1, mu_ydox1, UB1)

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
