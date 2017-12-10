import numpy as np
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy.stats import norm


def logistic_training(X,Y):
    clf = linear_model.LogisticRegression()
    clf.fit(X,Y)
    return clf

def inverse_logit(Z):
    return np.exp(Z) / (np.exp(Z)+1)

def intfy(W):
    return np.array(list(map(int,W)))

def causal_bound(p_obs,Hx,minmax_mode):
    def fun(x):
        if minmax_mode == 'min':
            return x
        elif minmax_mode == 'max':
            return -x

    def fun_deriv(x):
        if minmax_mode == 'min':
            return 1
        elif minmax_mode == 'max':
            return -1

    cons = ({'type': 'ineq',
             'fun': lambda x: -( p_obs * np.log(p_obs / x) + (1 - p_obs) * np.log((1 - p_obs) / (1 - x)) - Hx )},
            {'type': 'ineq',
             'fun': lambda x: (p_obs * np.log(p_obs / x) + (1 - p_obs) * np.log((1 - p_obs) / (1 - x)))}
            )

    lbounds = 0
    ubounds = 1
    bnds = [(lbounds, ubounds)]

    if minmax_mode == 'min':
        x0 = 1e-7
    elif minmax_mode == 'max':
        x0 = 1-1e-7

    res = minimize(fun, x0=x0,  constraints=cons, method='SLSQP',
                       bounds=bnds)

    if minmax_mode == 'min':
        return round(res.fun,3)
    elif minmax_mode == 'max':
        return round(-res.fun,3)


def logistic_proba(coef,care):
    coef = coef[0]
    if len(coef) > 1:
        x_coef = coef[0]
        z_coef = coef[1]
        x_care = care[0]
        z_care = care[1]
        result = (np.exp( x_coef * x_care + z_coef*z_care)) / (np.exp( x_coef * x_care + z_coef*z_care)+ 1)
    elif len(coef) == 1:
        z_coef = coef[0]
        z_care = care[0]
        result = (np.exp(z_coef * z_care)) / (np.exp(z_coef * z_care)+1)
    return result




print("----------------------START------------------------------")

np.random.seed(123)
N = 100000; Ns = 50
fig_version = 2


U1 = np.random.normal(20,10,N)
U2 = np.random.normal(-20,10,N)
U3 = np.random.binomial(1,0.8,N)

Z = U1 + U2
X = np.round((inverse_logit(2*U1 + 3*Z) + U3)/2,0)
# X = (X+1)/2

Y = np.round((inverse_logit(2*U2 - 3*Z) + (X + U3)/2 )/2,0)
# Y = (Y+1)/2

X = intfy(X)
Y = intfy(Y)

X_intv = np.asarray([0] * int(N/2) + [1] * int(N/2))
# Y_intv = X_intv * DZ * DU2 * DU3
# X_intv = (X_intv+1)/2
Y_intv = np.round((inverse_logit(2*U2 - 3*Z) + (X_intv + U3)/2 )/2,0)
# Y_intv = (Y_intv + 1)/2
X_intv = intfy(X_intv)
Y_intv = intfy(Y_intv)

X_obs = np.asarray(X)
Y_obs = np.asarray(Y)
Z_obs = np.asarray(Z)
Obs = pd.DataFrame({'X':X_obs, 'Y':Y_obs, 'Z':Z_obs})

X_intv = np.asarray(X_intv)
Y_intv = np.asarray(Y_intv)
Intv = pd.DataFrame({'X':X_intv,'Y':Y_intv, 'Z':Z_obs})

round_digit = 100

# Fix z
z_care = 0

if fig_version == 1:
    # Fix x
    x_care = 0
    care = [x_care, z_care]

    # Compute P(Y|x,z)
    ## Training the function $P(y|x,z) for all x,y,z
    obs_logit = logistic_training(Obs[['X','Z']],Obs['Y']) # P(Y|X,Z)
    ## Compute $P(Y=1 | x,z)
    p_obs = np.round( obs_logit.predict_proba(care)[0][1], round_digit)

    # Compute P(Y|do(x),z)
    ## Training the function $P(y|do(x),z) for all x,y,z
    true_logit = logistic_training(Intv[['X','Z']],Intv['Y'])
    ## Compute P(Y=1|do(x),z)
    p_true = np.round( true_logit.predict_proba(care)[0][1], round_digit)

    # Compute P(x|z)
    ## Training the function for P(X|Z)
    px_logit = logistic_training(Obs[['Z']],Obs['X'])
    ## Compute the function for P(x|z)
    ### Compute P(X|z)
    px_compute = px_logit.predict_proba(z_care)
    ### Choose P(x|z)
    pxz_cond = px_logit.predict_proba(z_care)[0][x_care]

    # Compute -log p(x)
    Hx = round(-np.log(pxz_cond),round_digit)

    # Compute P(x,z)
    ## Compute P(z)
    pz = np.round( norm.pdf(z_care, loc=np.mean(Obs['Z']), scale=np.std(Obs['Z'])  ),round_digit )
    ## Compute P(x,z)
    pxz = pxz_cond*pz

    # Compute P(x,y,z) for all y
    ## Compute P(Y=1, x, z)
    px_y1_z = np.round(pxz * p_obs, round_digit)
    ## Compute P(Y=0, x, z)
    px_y0_z = np.round(pxz * (1-p_obs), round_digit)


    # Analysis
    print("----------- Analysis --------")
    print("P(x|z): ", pxz_cond)
    print("P(z): ", pz)
    print("P(x,z): ",pxz)
    print("P(x,y,z) for y=1: ", px_y1_z)
    print("P(x,y,z) for y=0: ", px_y0_z)

    # Compute LB and UB
    LB = causal_bound(p_obs=p_obs, Hx=Hx, minmax_mode='min')
    UB = causal_bound(p_obs=p_obs, Hx=Hx, minmax_mode='max')
    print("X=", x_care, ", Interval:", LB, p_true, UB)
    print("p_obs:", p_obs, "Hx:", Hx)

    # Graphical illustration
    domain = np.linspace(0.0001, 0.9999, num=10000)
    output = p_obs * np.log(p_obs / domain) + (1 - p_obs) * np.log((1 - p_obs) / (1 - domain))
    plt.figure(1)
    plt.plot(domain, output)
    plt.plot(domain, [Hx] * len(domain))
    plt.axvline(p_true)

# Figure 2
elif fig_version == 2:
    ''' When X = 0 '''
    x_care = 0
    care = [x_care, z_care]
    # Compute P(Y|X=0,z)
    ## Training the function $P(y|x,z) for all x,y,z
    obs_logit = logistic_training(Obs[['X', 'Z']], Obs['Y'])  # P(Y|X,Z)
    ## Compute $P(Y=1 | X=0,z)
    p_obs0 = np.round(obs_logit.predict_proba(care)[0][1], round_digit)

    # Compute P(Y|do(X=0),z)
    ## Training the function $P(y|do(x),z) for all x,y,z
    true_logit = logistic_training(Intv[['X', 'Z']], Intv['Y'])
    ## Compute P(Y=1|do(X=0),z)
    p_true0 = np.round(true_logit.predict_proba(care)[0][1], round_digit)

    # Compute P(X=0|z)
    ## Training the function for P(X|Z) for all x,z
    px_logit = logistic_training(Obs[['Z']], Obs['X'])
    ## Compute the function for P(X=0|z)
    ### Compute P(X|z) for all x
    px_compute = px_logit.predict_proba(z_care)
    ### Choose P(X=0|z)
    pxz_cond = px_compute[0][x_care]

    # Compute -log p(X=0 | z)
    Hx0 = round(-np.log(pxz_cond), round_digit)

    # Compute P(X=0,z)
    ## Compute P(z)
    pz = np.round(norm.pdf(z_care, loc=np.mean(Obs['Z']), scale=np.std(Obs['Z'])), round_digit)
    ## Compute P(X=0,z)
    pxz = pxz_cond * pz

    # Compute P(X=0,y,z) for all y
    ## Compute P(Y=1, X=0, z)
    px_y1_z = np.round(pxz * p_obs0, round_digit)
    ## Compute P(Y=0, X=0, z)
    px_y0_z = np.round(pxz * (1 - p_obs0), round_digit)

    # Analysis for X=0, Z=z
    print("----------- Analysis for X=0, Z=z --------")
    print("P(X=0|z): ", pxz_cond)
    print("P(z): ", pz)
    print("P(X=0,z): ", pxz)
    print("P(X=0,Y=1,z) for y=1: ", px_y1_z)
    print("P(X=0,Y=0,z) for y=0: ", px_y0_z)

    # Compute LB and UB when X=0
    LB0 = causal_bound(p_obs=p_obs0, Hx=Hx0, minmax_mode='min')
    UB0 = causal_bound(p_obs=p_obs0, Hx=Hx0, minmax_mode='max')
    print("P(Y=1 | X=0, z):", p_obs0, "-log P(X=0):", Hx0)
    print("Interval:", np.round(LB0,round_digit), np.round(p_true0,round_digit), np.round(UB0,round_digit))





    ''' When X = 1 '''
    x_care = 1
    care = [x_care, z_care]
    # Compute P(Y|X=1,z)
    ## Training the function $P(y|x,z) for all x,y,z
    obs_logit = logistic_training(Obs[['X', 'Z']], Obs['Y'])  # P(Y|X,Z)
    ## Compute $P(Y=1 | X=1,z)
    p_obs1 = np.round(obs_logit.predict_proba(care)[0][1], round_digit)

    # Compute P(Y|do(X=1),z)
    ## Training the function $P(y|do(x),z) for all x,y,z
    true_logit = logistic_training(Intv[['X', 'Z']], Intv['Y'])
    ## Compute P(Y=1|do(X=1),z)
    p_true1 = np.round(true_logit.predict_proba(care)[0][1], round_digit)

    # Compute P(X=1|z)
    ## Training the function for P(X|Z) for all x,z
    px_logit = logistic_training(Obs[['Z']], Obs['X'])
    ## Compute the function for P(X=1|z)
    ### Compute P(X|z) for all x
    px_compute = px_logit.predict_proba(z_care)
    ### Choose P(X=1|z)
    pxz_cond = px_logit.predict_proba(z_care)[0][x_care]

    # Compute -log p(X=1 | z)
    Hx1 = round(-np.log(pxz_cond), round_digit)

    # Compute P(X=1,z)
    ## Compute P(z)
    pz = np.round(norm.pdf(z_care, loc=np.mean(Obs['Z']), scale=np.std(Obs['Z'])), round_digit)
    ## Compute P(X=1,z)
    pxz = pxz_cond * pz

    # Compute P(X=1,y,z) for all y
    ## Compute P(Y=1, X=1, z)
    px_y1_z = np.round(pxz * p_obs1, round_digit)
    ## Compute P(Y=0, X=1, z)
    px_y0_z = np.round(pxz * (1 - p_obs1), round_digit)

    # Analysis for X=1, Z=z
    print("----------- Analysis for X=1, Z=z --------")
    print("P(X=1|z): ", pxz_cond)
    print("P(z): ", pz)
    print("P(X=1,z): ", pxz)
    print("P(X=1,Y=0,z) for y=1: ", px_y1_z)
    print("P(X=1,Y=0,z) for y=0: ", px_y0_z)

    # Compute LB and UB when X=1
    LB1 = causal_bound(p_obs=p_obs1, Hx=Hx1, minmax_mode='min')
    UB1 = causal_bound(p_obs=p_obs1, Hx=Hx1, minmax_mode='max')
    print("P(Y=1 | X=1, z):", p_obs1, "-log P(X=1):", Hx1)
    print("Interval:", np.round(LB1, round_digit), np.round(p_true1, round_digit), np.round(UB1, round_digit))


    # Graphical illustration
    f, ax = plt.subplots(2, sharex=True)

    x_domain = np.array(range(0, 1001)) / 1000
    y_graphic = 0.4
    y_domain = np.array([y_graphic] * len(x_domain))

    ax[0].set_title('Interval of P(y|do(X= 0))')
    ax[0].plot(x_domain, y_domain)
    ax[0].axvline(x=0, ymin=0.48, ymax=0.52)
    ax[0].axvline(x=1, ymin=0.48, ymax=0.52)
    ax[0].axvline(x=LB0, ymin=0.35, ymax=0.65)
    ax[0].axvline(x=UB0, ymin=0.35, ymax=0.65)
    ax[0].plot(p_true0, y_graphic, marker='o',color='r')

    ax[1].plot(x_domain, y_domain)
    ax[1].set_title('Interval of P(y|do(X= 1))')
    ax[1].axvline(x=0, ymin=0.48, ymax=0.52)
    ax[1].axvline(x=1, ymin=0.48, ymax=0.52)
    ax[1].axvline(x=LB1, ymin=0.35, ymax=0.65)
    ax[1].axvline(x=UB1, ymin=0.35, ymax=0.65)
    ax[1].plot(p_true1, y_graphic, marker='o',color='r')

    print("----------------------END------------------------------")