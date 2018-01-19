import numpy as np
import scipy as sp
import pandas as pd
from sklearn.preprocessing import normalize


def intfy(W):
    return np.array(list(map(int,W)))

def inverse_logit(Z):
    return np.exp(Z) / (np.exp(Z)+1)


def gen_U(dim,N):
    mu1 = np.random.rand(dim)
    mu2 = np.random.rand(dim)
    mu3 = np.random.rand(dim)
    cov1 = np.dot(np.random.rand(dim, dim), np.random.rand(dim, dim).transpose())
    cov2 = np.dot(np.random.rand(dim, dim), np.random.rand(dim, dim).transpose())
    cov3 = np.dot(np.random.rand(dim, dim), np.random.rand(dim, dim).transpose())

    U1 = np.random.multivariate_normal(mu1, cov1, N)
    U2 = np.random.multivariate_normal(mu2, cov2, N)
    U3 = np.random.multivariate_normal(mu3, cov3, N)

    U1 = weird_projection(U1, 1)
    U2 = weird_projection(U2, 2)
    U3 = weird_projection(U3, 3)

    U1 = (U1 - np.mean(U1, axis=0)) / np.var(U1)
    U2 = (U2 - np.mean(U2, axis=0)) / np.var(U2)
    U3 = (U3 - np.mean(U3, axis=0)) / np.var(U3)

    return U1,U2,U3


def weird_projection(X,type_U):
    # X can be either matrix or vector
    if type_U == 1:
        X_proj = np.exp( np.sin(X) ) * np.log( np.tanh(X) ** 2 + 2) * np.exp(X)
    elif type_U == 2:
        X_proj = np.log( np.abs(X) ) * np.arctan(X ** 2)
    elif type_U == 3:
        X_proj = np.abs(X) ** 3 / np.exp(X)
    X_proj = (X_proj - np.mean(X_proj,axis=0))/np.var(X_proj)
    return X_proj


def gen_Z(dim, N, type_Z):
    U1,U2,U3 = gen_U(dim, N)
    if type_Z == 1:
        Z = U1 + U2
    elif type_Z == 2:
        U1 = np.sin(U1)
        U2 = np.log(U2 ** 2 + 1)
        Z = U1 * U2 
    elif type_Z == 3:
        Z = weird_projection(np.exp(U1) * np.log(abs(U2)+1),np.random.choice([1,2,3]))
    Z = ((Z - np.mean(Z,axis=0))/np.var(Z))
    return Z

def gen_X(dim, N, type_Z, type_X):
    coef_xz = np.reshape(1 * np.random.rand(dim), (dim, 1))
    coef_u1x = np.reshape(1 * np.random.rand(dim), (dim, 1))
    coef_u3x = np.reshape(1 * np.random.rand(dim), (dim, 1))

    U1, U2, U3 = gen_U(dim, N)
    Z = gen_Z(dim, N, type_Z)

    U1 = np.matrix(U1)
    U3 = np.matrix(U3)
    Z = np.matrix(Z)

    if type_X  == 1:
        X = U1*coef_u1x + U3*coef_u3x + Z*coef_xz
    elif type_X == 2:
        X = np.array( np.log(np.abs(U1*coef_u1x)+1) ) * np.array( np.sin(U1*coef_u1x) ) + np.array( np.exp(U3*coef_u3x) ) * np.array( np.abs(Z*coef_xz+1) )
    elif type_X == 3:
        X = weird_projection( np.array(np.log(np.abs(U1*coef_u1x)+1)), 1 ) + \
            weird_projection(  np.array( np.sin(U3*coef_u3x) ), 3) * \
            weird_projection( np.array( np.abs(Z*coef_xz+1) ), 2 )
    X = ((X - np.mean(X, axis=0)) / np.var(X))
    X = np.round( inverse_logit(X),0)
    X = intfy(X)
    X_intv = intfy(np.asarray([0] * int(N / 2) + [1] * int(N / 2)))
    return X,X_intv

def gen_Y(dim, N, type_Z, type_X, type_Y):
    U1, U2, U3 = gen_U(dim, N)
    Z = gen_Z(dim, N, type_Z)
    X,X_intv = gen_X(dim, N, type_Z, type_X)

    coef_zy = np.reshape(1 * np.random.rand(dim), (dim, 1))
    coef_u2y = np.reshape(1 * np.random.rand(dim), (dim, 1))
    coef_u3y = np.reshape(1 * np.random.rand(dim), (dim, 1))

    U2 = np.matrix(U2)
    U3 = np.matrix(U3)
    Z = np.matrix(Z)
    X = np.matrix(X)
    X_intv = np.matrix(X_intv)

    if type_Y  == 1:
        Y = U2 + U3 + X + Z
        Y_intv = U2 + U3 + X_intv + Z
    elif type_Y == 2:
        Y = np.array( np.sin(U2*coef_u2y) ) * \
            np.array( np.array(np.tanh(U3*coef_u3y)) * ( np.exp( np.array(X.T)  ) ) )* \
            np.array( np.abs(Z*coef_zy+1)  )

        Y_intv = np.array(np.sin(U2 * coef_u2y)) * \
            np.array(np.array(np.tanh(U3 * coef_u3y)) * (np.exp(np.array(X_intv.T) ))) * \
            np.array(np.abs(Z * coef_zy + 1))

    elif type_Y == 3:
        Y = weird_projection(np.array( U2*coef_u2y) ,np.random.choice([1,2,3])) * \
            weird_projection(np.array( U3*coef_u3y),np.random.choice([1,2,3])) * \
            np.array( np.sin(np.array(X.T)) ) * np.array( np.exp(np.array(Z*coef_zy)) )

        Y_intv = weird_projection(np.array(U2 * coef_u2y), np.random.choice([1, 2, 3])) * \
            weird_projection(np.array(U3 * coef_u3y), np.random.choice([1, 2, 3])) * \
            np.array(np.sin(np.array(X_intv.T))) * np.array(np.exp(np.array(Z * coef_zy)))

    # Y = ((Y - np.mean(Y, axis=0)) / np.var(Y))
    # Y_intv = ((Y_intv - np.mean(Y_intv, axis=0)) / np.var(Y_intv))
    return Y, Y_intv

def str_data(dim, N, Ns, type_Z, type_X, type_Y):
    Z = gen_Z(dim, N, type_Z)
    X, X_intv = gen_X(dim, N, type_Z, type_X)
    Y, Y_intv = gen_Y(dim, N, type_Z, type_X, type_Y)

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

    X_intv = np.asarray(X_intv)
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




# if __name__ == '__main__':

dim = 5
N = 1000
Ns = 300
type_Z = 1
type_X = 3
type_Y = 3

Obs,Intv,Intv_S = str_data(dim,N,Ns,type_Z,type_X,type_Y)

