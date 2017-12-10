import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

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


np.random.seed(12345)
N = 100000; Ns = 50
fig_version = 2

U1 = np.random.normal(1,10,N)
U2 = np.random.normal(0,10,N)
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

#
# X_intv = np.asarray([0] * int(N/2) + [1] * int(N/2))
# X_intv = np.random.permutation(X_intv)
#
# Y_intv = np.round((inverse_logit(2*U2 - 3*Z) + (X_intv + U3)/2 )/2,0)
# # Y_intv = (Y+1)/2
# Y_intv = intfy(Y_intv)

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

print("----------- Analysis --------")
print("P(X=0,Y=0)",len(Obs[(Obs['X']==0)&(Obs['Y']==0)])/N)
print("P(X=0,Y=1)",len(Obs[(Obs['X']==0)&(Obs['Y']==1)])/N)
print("P(X=1,Y=0)",len(Obs[(Obs['X']==1)&(Obs['Y']==0)])/N)
print("P(X=1,Y=1)",len(Obs[(Obs['X']==1)&(Obs['Y']==1)])/N)
print("P(X=0)", len(Obs[Obs['X']==0])/N)
print("P(X=1)", len(Obs[Obs['X']==1])/N)







# Figure version 1
if fig_version == 1:
    x_care = 0
    p = len(Intv[(Intv['X']==x_care)&(Intv['Y']==1)]) /len(Intv[(Intv['X'] == x_care)])
    p_obs = len(Obs[(Obs['X']==x_care)&(Obs['Y']==1)]) /len(Obs[(Obs['X'] == x_care)])
    px = len(Obs[Obs['X'] == x_care]) / N

    Hx = -np.log(px)

    LB = causal_bound(p_obs=p_obs,Hx=Hx,minmax_mode='min')
    UB = causal_bound(p_obs=p_obs,Hx=Hx,minmax_mode='max')
    print("X=",x_care, ", Interval:", LB,p,UB)
    print("p_obs:",p_obs, "Hx:", Hx )

    domain = np.linspace(0.0001,0.9999,num=10000)
    output = p_obs*np.log(p_obs/domain) + (1-p_obs)*np.log((1-p_obs)/(1-domain))

    plt.figure(1)
    plt.plot(domain,output)
    plt.plot(domain,[Hx]*len(domain))
    plt.axvline(p)

elif fig_version == 2:
    f, ax = plt.subplots(2, sharex=True)

    x_care = 0
    p = len(Intv[(Intv['X'] == x_care) & (Intv['Y'] == 1)]) / len(Intv[(Intv['X'] == x_care)])
    p_obs = len(Obs[(Obs['X'] == x_care) & (Obs['Y'] == 1)]) / len(Obs[(Obs['X'] == x_care)])
    px = len(Obs[Obs['X'] == x_care]) / N

    Hx = -np.log(px)

    LB0 = causal_bound(p_obs=p_obs, Hx=Hx, minmax_mode='min')
    UB0 = causal_bound(p_obs=p_obs, Hx=Hx, minmax_mode='max')
    print("X=", x_care, ", Interval:", LB0, p, UB0)
    print("p_obs:", p_obs, "Hx:", Hx)

    x_domain = np.array(range(0,101))/100
    y0 = 0.4
    y_domain_x0 = np.array([y0]*len(x_domain))

    ax[0].plot(x_domain, y_domain_x0)
    ax[0].set_title('Interval of P(y|do(X= 0))')
    ax[0].axvline(x=0,ymin= 0.48, ymax= 0.52)
    ax[0].axvline(x=1,ymin= 0.48, ymax= 0.52)
    ax[0].axvline(x=LB0,ymin=0.35, ymax=0.65)
    ax[0].axvline(x=UB0,ymin=0.35, ymax=0.65)
    ax[0].plot(p,y0,'o')


    x_care = 1
    p1 = len(Intv[(Intv['X']==x_care)&(Intv['Y']==1)]) /len(Intv[(Intv['X'] == x_care)])
    p_obs = len(Obs[(Obs['X']==x_care)&(Obs['Y']==1)]) /len(Obs[(Obs['X'] == x_care)])
    px = len(Obs[Obs['X'] == x_care]) / N
    Hx = -np.log(px)

    LB1 = causal_bound(p_obs,Hx ,'min')
    UB1 = causal_bound(p_obs,Hx ,'max')
    print("X=",x_care, ", Interval:", LB1,p1,UB1)
    print("p_obs:",p_obs, "Hx:", Hx )

    x_domain = np.array(range(0,101))/100
    y1 = 0.4
    y_domain_x1 = np.array([y1]*len(x_domain))

    ax[1].plot(x_domain, y_domain_x1)
    ax[1].set_title('Interval of P(y|do(X= 1))')
    ax[1].axvline(x=0,ymin= 0.48, ymax= 0.52)
    ax[1].axvline(x=1,ymin= 0.48, ymax= 0.52)
    ax[1].axvline(x=LB1,ymin=0.35, ymax=0.65)
    ax[1].axvline(x=UB1,ymin=0.35, ymax=0.65)
    ax[1].plot(p1,y1,'o')

print("----------------------END------------------------------")