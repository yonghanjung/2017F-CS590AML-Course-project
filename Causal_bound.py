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

def linear_training(X,Y):
    clf = linear_model.LinearRegression()
    clf.fit(X, Y)
    return clf

def inverse_logit(Z):
    return np.exp(Z) / (np.exp(Z)+1)

def intfy(W):
    return np.array(list(map(int,W)))

def causal_bound_case1(p_obs,Hx,minmax_mode):
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
    if p_obs < 1 and p_obs > 0:
        cons = ({'type': 'ineq',
                 'fun': lambda x: -( p_obs * np.log(p_obs / x) + (1 - p_obs) * np.log((1 - p_obs) / (1 - x)) - Hx )},
                {'type': 'ineq',
                 'fun': lambda x: (p_obs * np.log(p_obs / x) + (1 - p_obs) * np.log((1 - p_obs) / (1 - x)))}
                )
    elif p_obs == 1:
        cons = ({'type': 'ineq',
                 'fun': lambda x: -(p_obs * np.log(p_obs / x)  - Hx)},
                {'type': 'ineq',
                 'fun': lambda x: (p_obs * np.log(p_obs / x) )}
                )
    elif p_obs == 0:
        cons = ({'type': 'ineq',
                 'fun': lambda x: -((1 - p_obs) * np.log((1 - p_obs) / (1 - x)) - Hx)},
                {'type': 'ineq',
                 'fun': lambda x: ((1 - p_obs) * np.log((1 - p_obs) / (1 - x)))}
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


def Causal_bound(case_num, dim, Obs, Intv, Intv_S, x, z):
    if case_num == 1:
        x_care = x
        N = len(Obs)
        p_true = len(Intv[(Intv['X'] == x_care) & (Intv['Y'] == 1)]) / len(Intv[(Intv['X'] == x_care)])
        p_obs = len(Obs[(Obs['X'] == x_care) & (Obs['Y'] == 1)]) / len(Obs[(Obs['X'] == x_care)])
        px = len(Obs[Obs['X'] == x_care]) / N

        Hx = -np.log(px)

        LB = causal_bound_case1(p_obs=p_obs, Hx=Hx, minmax_mode='min')
        UB = causal_bound_case1(p_obs=p_obs, Hx=Hx, minmax_mode='max')

        return LB, p_true,  UB
    elif case_num == 11:
        x_care = x
        z_care = z
        round_digit = 20
        care = [x_care, z_care]

        # Compute P(Y|x,z)
        ## Training the function $P(y|x,z) for all x,y,z
        obs_logit = logistic_training(Obs[['X', 'Z']], Obs['Y'])  # P(Y|X,Z)
        ## Compute $P(Y=1 | x,z)
        p_obs = np.round(obs_logit.predict_proba(care)[0][1], round_digit)

        # Compute P(Y|do(x),z)
        ## Training the function $P(y|do(x),z) for all x,y,z
        true_logit = logistic_training(Intv[['X', 'Z']], Intv['Y'])
        ## Compute P(Y=1|do(x),z)
        p_true = np.round(true_logit.predict_proba(care)[0][1], round_digit)

        # Compute P(x|z)
        ## Training the function for P(X|Z)
        px_logit = logistic_training(Obs[['Z']], Obs['X'])
        ## Compute the function for P(x|z)
        ### Compute P(X|z)
        px_compute = px_logit.predict_proba(z_care)[0]
        ### Choose P(x|z)
        pxz_cond = px_compute[x_care]

        # Compute -log p(x)
        Hx = round(-np.log(pxz_cond), round_digit)

        LB = causal_bound_case1(p_obs=p_obs, Hx=Hx, minmax_mode='min')
        UB = causal_bound_case1(p_obs=p_obs, Hx=Hx, minmax_mode='max')
        return LB, p_true, UB

    elif case_num == 3:
        x_care = x
        N = len(Obs)

        # Compute mu, std of P(Y=y | x)
        Obs_yx = Obs[Obs['X'] == x_care]['Y']
        mu_yx = np.mean(Obs_yx)
        std_yx = np.std(Obs_yx)

        # Compute mu of P(Y=y | do(x))
        Intv_yx = Intv[Intv['X'] == x_care]['Y']
        mu_ydox = np.mean(Intv_yx)

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

        return LB, mu_ydox, UB

    elif case_num == 31:
        x_care = x
        z_care = z
        care = [x_care, z_care]

        # Compute mu and std of P(Y|x,z)
        ## Training for P(Y|x,z) for all x and z
        obs_lin = linear_training(Obs[['X', 'Z']], Obs['Y'])
        ### Compute the residual distribution for std
        obs_result = obs_lin.predict(Obs[['X', 'Z']])
        obs_resid = obs_result - Obs['Y']
        ## Compute mu and std of P(Y|x,z)
        mu_yxz = obs_lin.predict(care)[0]
        std_yxz = np.std(obs_resid)

        # Compute mu and std of Ds(Y|do(x),z)
        ## Training for D(Y|do(x),z) for all x and z
        Sdox_lin = linear_training(Intv_S[['X', 'Z']], Intv_S['Y'])
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
        pxz_logit = logistic_training(Obs[['Z']], Obs['X'])
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

        return LB, mu_ydox, UB

    elif case_num == 4:
        # Fix X=x
        x_care = x
        N = len(Obs)

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

        return LB, mu_ydox, UB

    elif case_num == 41:
        x_care = x
        z_care = z
        care = [x_care] + z_care

        # Compute mu and std of P(Y|x,z)
        ## Training for P(Y|x,z) for all x and z
        obs_lin = linear_training(Obs[['X'] + list(range(dim))], Obs['Y'])
        ### Compute the residual distribution for std
        obs_result = obs_lin.predict(Obs[['X'] + list(range(dim))])
        obs_resid = obs_result - Obs['Y']
        ## Compute mu and std of P(Y|x,z)
        mu_yxz = obs_lin.predict(care)[0]
        std_yxz = np.std(obs_resid)

        # Compute mu and std of Ds(Y|do(x),z)
        ## Training for D(Y|do(x),z) for all x and z
        Sdox_lin = linear_training(Intv_S[['X'] + list(range(dim))], Intv_S['Y'])
        ### Compute the residual distribution for std
        Sdox_resid = Sdox_lin.predict(Intv_S[['X'] + list(range(dim))]) - Intv_S['Y']
        ## Compute mu and std of Ds(Y|do(x),z)
        mu_sydox = Sdox_lin.predict(care)[0]
        std_sydox = np.std(Sdox_resid)

        # Compute mu and std of P(Y|do(x),z)
        ## Training for P(Y|do(x),z) for all x and z
        dox_lin = linear_training(Intv[['X'] + list(range(dim))], Intv['Y'])
        ### Compute the residual distribution for std
        dox_resid = dox_lin.predict(Intv[['X'] + list(range(dim))]) - Intv['Y']
        ## Compute mu and std of P(Y|do(x),z)
        mu_ydox = dox_lin.predict(care)[0]
        std_ydox = np.std(dox_resid)
        # std_sydox = std_ydox

        # Compute P(X=x_care | z)
        ## Training P(X|Z) for all z and x
        pxz_logit = logistic_training(Obs[list(range(dim))], Obs['X'])
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
        return LB, mu_ydox, UB