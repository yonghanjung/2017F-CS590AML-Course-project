import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

def intfy(W):
    return np.array(list(map(int,W)))
def inverse_logit(Z):
    return np.exp(Z) / (np.exp(Z)+1)

def Data_generation(case_num, N, Ns, dim, seed_num = 123):
    if case_num == 1 or case_num == 11:
        np.random.seed(seed_num)

        U1 = np.random.normal(1, 10, N)
        U2 = np.random.normal(0, 10, N)
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
        Y_sintv = Y_intv[sample_indices]
        Z_sintv = Z_obs[sample_indices]
        Intv_S = pd.DataFrame({'X': X_sintv, 'Y': Y_sintv, 'Z': Z_sintv})

        return Obs, Intv, Intv_S

    elif case_num == 3 or case_num == 31:
        np.random.seed(seed_num)
        U1 = np.random.normal(0, 20, N)
        U2 = np.random.normal(0, 20, N)
        U3 = np.random.normal(0, 20, N)

        Z = U1 + U2
        X = intfy(np.round(inverse_logit(1 * Z - 1 * U1 + 1 * U3), 0))
        Y = Z + U2 + U3 + 10 * X

        X_intv = intfy(np.asarray([0] * int(N / 2) + [1] * int(N / 2)))
        Y_intv = Z + U2 + U3 + 10 * X_intv

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

        # '''For Lin UCB'''
        # np.random.seed(seed_num)
        #
        # mu1 = np.random.rand(dim)
        # mu2 = np.random.rand(dim)
        # mu3 = np.random.rand(dim)
        # cov1 = np.dot(np.random.rand(dim, dim), np.random.rand(dim, dim).transpose())
        # cov2 = np.dot(np.random.rand(dim, dim), np.random.rand(dim, dim).transpose())
        # cov3 = np.dot(np.random.rand(dim, dim), np.random.rand(dim, dim).transpose())
        #
        # coef_xz = np.reshape(1 * np.random.rand(dim), (dim, 1))
        # coef_u1x = np.reshape(1 * np.random.rand(dim), (dim, 1))
        # coef_u3x = np.reshape(1 * np.random.rand(dim), (dim, 1))
        # coef_zy = np.reshape(1 * np.random.rand(dim), (dim, 1))
        # coef_u2y = np.reshape(1 * np.random.rand(dim), (dim, 1))
        # coef_u3y = np.reshape(1 * np.random.rand(dim), (dim, 1))
        #
        # U1 = np.random.multivariate_normal(mu1, cov1, N)
        # U2 = np.random.multivariate_normal(mu2, cov2, N)
        # U3 = np.random.multivariate_normal(mu3, cov3, N)
        #
        # Z = normalize(1 * U1 + 1 * U2)
        # sample_indices_disc = np.random.choice(list(range(0, dim)), int(dim / 3), replace=False)
        # Z_replace = np.random.binomial(1, 0.5, (N, len(sample_indices_disc)))
        # Z[:, sample_indices_disc] = Z_replace
        #
        # X = intfy(np.round(
        #     inverse_logit(normalize(np.matrix(Z) * coef_xz + np.matrix(U1) * coef_u1x + np.matrix(U3) * coef_u3x)), 0))
        #
        # Y_orig = np.array(np.matrix(Z) * coef_zy) + \
        #          np.array(np.matrix(U2) * coef_u2y) + \
        #          np.array(np.matrix(U3) * coef_u3y)
        #
        # Y_norm = (Y_orig - np.mean(Y_orig) )/np.std(Y_orig)
        # Y = 1*Y_norm + np.reshape(1 * X, (N, 1))
        #
        # X_intv = intfy(np.asarray([0] * int(N / 2) + [1] * int(N / 2)))
        # Y_intv = 1*Y_norm + np.reshape(1 * X_intv, (N, 1))

        ########################################################################
        ########################################################################
        '''For Lin UCB'''
        np.random.seed(seed_num)

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

        Z = normalize(-1 * U1 + 1 * U2)
        sample_indices_disc = np.random.choice(list(range(0, dim)), int(dim / 3), replace=False)
        Z_replace = np.random.binomial(1, 0.5, (N, len(sample_indices_disc)))
        Z[:, sample_indices_disc] = Z_replace

        X = intfy(np.round(
            inverse_logit(normalize(np.matrix(Z) * coef_xz + np.matrix(U1) * coef_u1x + np.matrix(U3) * coef_u3x)), 0))

        Y_orig = np.array(np.matrix(Z) * coef_zy) + \
                 np.array(np.matrix(U2) * coef_u2y) + \
                 np.array(np.matrix(U3) * coef_u3y)

        Y_norm = (Y_orig - np.mean(Y_orig) )/np.std(Y_orig)
        # Y_norm = Y_orig
        Y = 3*Y_norm + np.reshape(1 * X, (N, 1))

        X_intv = intfy(np.asarray([0] * int(N / 2) + [1] * int(N / 2)))
        Y_intv = 3*Y_norm + np.reshape(1 * X_intv, (N, 1))

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
