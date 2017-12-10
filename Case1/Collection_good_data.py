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

'''Example 1 '''
np.random.seed(1)
N = 50000; Ns = 50

U1 = np.random.normal(-500,10,N)
U2 = np.random.normal(-100,10,N)
U3 = np.random.binomial(1,0.8,N)

Z = U1 + U2
X = (2*np.round(inverse_logit(3*Z - 2*U1),0)-1) * (2*U3-1)
X = (X+1)/2

Y = (2*X-1) * (2*np.round(inverse_logit(2*Z - 3*U2),0)-1) * (2*U3-1)
Y = (Y+1)/2

X = intfy(X)
Y = intfy(Y)

''' Example 2 '''

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


''' Example 3 '''


U1 = np.random.normal(10,10,N)
U2 = np.random.normal(-10,10,N)
U3 = np.random.binomial(1,0.1,N)

Z = U1 + U2
DZ = 2*(Z > 0)-1
DU1 = 2*(U1 > 0)-1
DU2 = 2*(U2 > 0)-1
DU3 = 2*U3-1

X = DZ * DU1 * DU3
Y = X * DZ * DU2 * DU3

X = (X+1)/2
Y = (Y+1)/2
X = intfy(X)
Y = intfy(Y)