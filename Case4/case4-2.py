import numpy as np
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy.stats import norm
from sklearn.preprocessing import normalize


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

Y_norm = (Y_orig - np.mean(Y_orig))/np.std(Y_orig)
Y = intfy( np.round( inverse_logit( Y_norm + np.reshape(1*X,(N,1)) ), 0) )

X_intv = intfy(np.asarray([0] * int(N / 2) + [1] * int(N / 2)))
Y_intv = intfy( np.round( inverse_logit( Y_norm + np.reshape(1*X_intv,(N,1)) ), 0) )
#
# mu1 = np.random.rand(dim)
# mu2 = np.random.rand(dim)
# mu3 = np.random.rand(dim)
# cov1 = np.dot( np.random.rand(dim,dim), np.random.rand(dim,dim).transpose() )
# cov2 = np.dot( np.random.rand(dim,dim), np.random.rand(dim,dim).transpose() )
# cov3 = np.dot( np.random.rand(dim,dim), np.random.rand(dim,dim).transpose() )
#
# coef_xz = np.reshape(1*np.random.rand(dim), (dim,1))
# coef_u1x = np.reshape(1*np.random.rand(dim), (dim,1))
# coef_u3x = np.reshape(1*np.random.rand(dim), (dim,1))
# coef_zy = np.reshape(1*np.random.rand(dim), (dim,1))
#
# U1 = np.random.multivariate_normal(mu1,cov1,N)
# U2 = np.random.multivariate_normal(mu2,cov2,N)
# U3 = np.random.multivariate_normal(mu3,cov3,N)
#
#
#
# Z = normalize( U1+U2 )
# sample_indices_disc = np.random.choice(list(range(0,dim)),int(dim/3),replace=False)
# Z_replace = np.random.binomial(1,0.5,(N,len(sample_indices_disc)))
# Z[:,sample_indices_disc] = Z_replace
#
# X = intfy( np.round( inverse_logit( normalize( np.matrix(Z)*coef_xz + np.matrix(U1) * coef_u1x + np.matrix(U3) * coef_u3x ) ), 0) )
# Y = np.array(np.matrix(Z)*coef_zy) + \
#     np.reshape(np.array(U2[:,dim-1]),(N,1)) + \
#     np.reshape(np.array(U3[:,1]),(N,1)) + \
#     1*np.reshape(X,(N,1))
#
# X_intv = intfy( np.asarray([0] * int(N/2) + [1] * int(N/2)) )
# Y_intv = np.array(np.matrix(Z)*coef_zy) + \
#     np.reshape(np.array(U2[:,dim-1]),(N,1)) + \
#     np.reshape(np.array(U3[:,1]),(N,1)) + \
#     1*np.reshape(X_intv,(N,1))

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

# z_care = list( np.random.multivariate_normal( np.mean(Z,axis=0), np.diag( np.std(Z,axis=0) ), 1 )[0] )
z_care = list(np.mean(Z,axis=0) )
# z_care = list(Z[0]+2)
round_digit = 10

if fig_version == 1:
    # Fix x
    x_care = 0
    care = [x_care] + z_care

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
    care = [x_care] + z_care
    # Compute P(Y|X=0,z)
    ## Training the function $P(y|x,z) for all x,y,z
    obs_logit = logistic_training(Obs[ ['X']+list(range(dim)) ], Obs['Y'])  # P(Y|X,Z)
    ## Compute $P(Y=1 | X=0,z)
    p_obs0 = np.round(obs_logit.predict_proba(care)[0][1], round_digit)

    # Compute P(Y|do(X=0),z)
    ## Training the function $P(y|do(x),z) for all x,y,z
    true_logit = logistic_training(Intv[ ['X']+list(range(dim)) ], Intv['Y'])
    ## Compute P(Y=1|do(X=0),z)
    p_true0 = np.round(true_logit.predict_proba(care)[0][1], round_digit)

    # Compute P(X=0|z)
    ## Training the function for P(X|Z) for all x,z
    px_logit = logistic_training(Obs[list(range(dim))], Obs['X'])
    ## Compute the function for P(X=0|z)
    ### Compute P(X|z) for all x
    px_compute = px_logit.predict_proba(z_care)
    ### Choose P(X=0|z)
    pxz_cond = px_compute[0][x_care]

    # Compute -log p(X=0 | z)
    Hx0 = round(-np.log(pxz_cond), round_digit)

    # # Compute P(X=0,z)
    # ## Compute P(z)
    # pz = np.round(norm.pdf(z_care, loc=np.mean(Obs['Z']), scale=np.std(Obs['Z'])), round_digit)
    # ## Compute P(X=0,z)
    # pxz = pxz_cond * pz
    #
    # # Compute P(X=0,y,z) for all y
    # ## Compute P(Y=1, X=0, z)
    # px_y1_z = np.round(pxz * p_obs0, round_digit)
    # ## Compute P(Y=0, X=0, z)
    # px_y0_z = np.round(pxz * (1 - p_obs0), round_digit)

    # Analysis for X=0, Z=z
    # print("----------- Analysis for X=0, Z=z --------")
    # print("P(X=0|z): ", pxz_cond)
    # print("P(z): ", pz)
    # print("P(X=0,z): ", pxz)
    # print("P(X=0,Y=1,z) for y=1: ", px_y1_z)
    # print("P(X=0,Y=0,z) for y=0: ", px_y0_z)

    # Compute LB and UB when X=0
    LB0 = causal_bound(p_obs=p_obs0, Hx=Hx0, minmax_mode='min')
    UB0 = causal_bound(p_obs=p_obs0, Hx=Hx0, minmax_mode='max')
    print("P(Y=1 | X=0, z):", p_obs0, "-log P(X=0):", Hx0)
    print("Interval:", np.round(LB0,round_digit), np.round(p_true0,round_digit), np.round(UB0,round_digit))





    ''' When X = 1 '''
    x_care = 1
    care = [x_care] + z_care
    # Compute P(Y|X=1,z)
    ## Training the function $P(y|x,z) for all x,y,z
    # obs_logit = logistic_training(Obs[['X'] + list(range(dim))], Obs['Y'])  # P(Y|X,Z)
    ## Compute $P(Y=1 | X=1,z)
    p_obs1 = np.round(obs_logit.predict_proba(care)[0][1], round_digit)

    # Compute P(Y|do(X=1),z)
    ## Training the function $P(y|do(x),z) for all x,y,z
    # true_logit = logistic_training(Intv[['X', 'Z']], Intv['Y'])
    ## Compute P(Y=1|do(X=1),z)
    p_true1 = np.round(true_logit.predict_proba(care)[0][1], round_digit)

    # Compute P(X=1|z)
    ## Training the function for P(X|Z) for all x,z
    # px_logit = logistic_training(Obs[['Z']], Obs['X'])
    ## Compute the function for P(X=1|z)
    ### Compute P(X|z) for all x
    px_compute = px_logit.predict_proba(z_care)
    ### Choose P(X=1|z)
    pxz_cond = px_logit.predict_proba(z_care)[0][x_care]

    # Compute -log p(X=1 | z)
    Hx1 = round(-np.log(pxz_cond), round_digit)

    # # Compute P(X=1,z)
    # ## Compute P(z)
    # pz = np.round(norm.pdf(z_care, loc=np.mean(Obs['Z']), scale=np.std(Obs['Z'])), round_digit)
    # ## Compute P(X=1,z)
    # pxz = pxz_cond * pz
    #
    # # Compute P(X=1,y,z) for all y
    # ## Compute P(Y=1, X=1, z)
    # px_y1_z = np.round(pxz * p_obs1, round_digit)
    # ## Compute P(Y=0, X=1, z)
    # px_y0_z = np.round(pxz * (1 - p_obs1), round_digit)
    #
    # # Analysis for X=1, Z=z
    # print("----------- Analysis for X=1, Z=z --------")
    # print("P(X=1|z): ", pxz_cond)
    # print("P(z): ", pz)
    # print("P(X=1,z): ", pxz)
    # print("P(X=1,Y=0,z) for y=1: ", px_y1_z)
    # print("P(X=1,Y=0,z) for y=0: ", px_y0_z)

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