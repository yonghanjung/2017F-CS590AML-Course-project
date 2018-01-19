import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

class Data_generaiton(object):
    np.random.seed(12345)
    def __init__(self, num_obs, num_intv, dim):
        self.num_obs = num_obs
        self.num_intv = num_intv
        self.dim = dim

    def intfy(self,W):
        return np.array(list(map(int, W)))

    def inverse_logit(self,Z):
        return np.exp(Z) / (np.exp(Z) + 1)

    def weird_projection(self, X):
        # X can be either matrix or vector
        X_proj = np.log(np.abs(X)+20) * np.arctan(X ** 2) + np.exp(np.sin(X)) - np.log(np.tanh(X) ** 2 + 2) * np.exp(X)
        X_proj = (X_proj - np.mean(X_proj, axis=0)) / np.var(X_proj)
        return X_proj

    def gen_U(self):
        mu1 = np.random.rand(self.dim)
        mu2 = np.random.rand(self.dim)
        mu3 = np.random.rand(self.dim)
        cov1 = np.dot(np.random.rand(self.dim, self.dim),
                      np.random.rand(self.dim, self.dim).transpose())
        cov2 = np.dot(np.random.rand(self.dim, self.dim),
                      np.random.rand(self.dim, self.dim).transpose())
        cov3 = np.dot(np.random.rand(self.dim, self.dim),
                      np.random.rand(self.dim, self.dim).transpose())

        U1 = np.random.multivariate_normal(mu1, cov1, self.num_obs)
        U2 = np.random.multivariate_normal(mu2, cov2, self.num_obs)
        U3 = np.random.multivariate_normal(mu3, cov3, self.num_obs)

        U1 = self.weird_projection(U1)
        U2 = self.weird_projection(U2)
        U3 = self.weird_projection(U3)

        U1 = (U1 - np.mean(U1, axis=0)) / np.var(U1)
        U2 = (U2 - np.mean(U2, axis=0)) / np.var(U2)
        U3 = (U3 - np.mean(U3, axis=0)) / np.var(U3)

        self.U1 = U1
        self.U2 = U2
        self.U3 = U3

    def gen_Z(self):
        Z = np.exp(self.U1) + np.exp(self.U2)
        self.Z = ((Z - np.mean(Z, axis=0)) / np.var(Z))

    def gen_X(self):
        coef_xz = np.reshape(-1*np.random.rand(self.dim), (self.dim, 1))
        coef_u1x = np.reshape(1*np.random.rand(self.dim), (self.dim, 1))
        coef_u3x = np.reshape(-1*np.random.rand(self.dim), (self.dim, 1))

        U1 = np.matrix(self.U1)
        U3 = np.matrix(self.U3)
        Z = np.matrix(self.Z)

        X = np.exp(U1 * coef_u1x ) - (U3 * coef_u3x ) + np.abs(Z * coef_xz )

        X = ((X - np.mean(X, axis=0)) / np.var(X))
        # print(X)
        X = np.round(self.inverse_logit(X), 0)
        self.X_obs = self.intfy(X)
        self.X_intv = self.intfy(np.asarray([0] * int(self.num_obs / 2) +
                                            [1] * int(self.num_obs / 2)))

    def gen_Y(self):
        coef_zy = np.reshape(1 * np.random.rand(self.dim), (self.dim, 1))
        coef_u2y = np.reshape(1 * np.random.rand(self.dim), (self.dim, 1))
        coef_u3y = np.reshape(1 * np.random.rand(self.dim), (self.dim, 1))

        U2 = np.matrix(self.U2)
        U3 = np.matrix(self.U3)
        Z = np.matrix(self.Z)
        X_obs = np.matrix(self.X_obs)
        X_intv = np.matrix(self.X_intv)

        Y = np.array(np.sin(U2 * coef_u2y)) * \
            np.array(-1*np.array(-200*np.tanh(U3 * coef_u3y)) +
                     (np.array(X_obs.T))) * \
            np.array(np.abs(Z * coef_zy + 1))

        self.Y = 100*((Y - np.mean(Y, axis=0)) / np.var(Y))

        Y_intv = np.array(np.sin(U2 * coef_u2y)) * \
            np.array(-1 * np.array(-200 * np.tanh(U3 * coef_u3y)) +
                     (np.array(X_intv.T))) * \
            np.array(np.abs(Z * coef_zy + 1))

        self.Y_intv = 100*((Y_intv - np.mean(Y, axis=0)) / np.var(Y))

    def structure_data(self):
        X = self.X_obs
        X_intv = self.X_intv
        Y = self.Y
        Y_intv = self.Y_intv
        Z = self.Z

        X_obs = np.asarray(X)
        Y_obs = np.asarray(Y)
        Z_obs = np.asarray(Z)

        Obs_X = pd.DataFrame(X_obs)
        Obs_Y = pd.DataFrame(Y_obs)
        Obs_Z = pd.DataFrame(Z_obs)

        Obs_XY = pd.concat([Obs_X, Obs_Y], axis=1)
        Obs_XY.columns = ['X', 'Y']
        Obs_Z.columns = range(self.dim)
        self.Obs = pd.concat([Obs_XY, Obs_Z], axis=1)


        X_intv = np.asarray(X_intv)
        Y_intv = np.asarray(Y_intv)
        Intv_X = pd.DataFrame(X_intv)
        Intv_Y = pd.DataFrame(Y_intv)
        Intv_Z = pd.DataFrame(Z_obs)

        Intv_XY = pd.concat([Intv_X, Intv_Y], axis=1)
        Intv_XY.columns = ['X', 'Y']
        Intv_Z.columns = range(self.dim)
        self.Intv = pd.concat([Intv_XY, Intv_Z], axis=1)

        sample_indces_x1 = np.random.choice(list(range(0, int(self.num_obs / 2))), int(self.num_intv / 2), replace=False)
        sample_indces_x0 = np.random.choice(list(range(int(self.num_obs / 2), self.num_obs)), int(self.num_intv / 2), replace=False)
        sample_indices = np.asarray(list(sample_indces_x1) + list(sample_indces_x0))

        X_sintv = X_intv[sample_indices]
        Y_sintv = Y_intv[sample_indices]
        Z_sintv = Z_obs[sample_indices]
        SIntv_X = pd.DataFrame(X_sintv)
        SIntv_Y = pd.DataFrame(Y_sintv)
        SIntv_Z = pd.DataFrame(Z_sintv)

        SIntv_XY = pd.concat([SIntv_X, SIntv_Y], axis=1)
        SIntv_XY.columns = ['X', 'Y']
        SIntv_Z.columns = range(self.dim)
        self.Intv_S = pd.concat([SIntv_XY, SIntv_Z], axis=1)

    def data_gen(self):
        self.gen_U()
        self.gen_Z()
        self.gen_X()
        self.gen_Y()
        self.structure_data()





# num_obs = 100
# num_intv = 10
# dim = 5
# gen_data = Data_generaiton(num_obs, num_intv, dim)
# gen_data.data_gen()
#
# X = gen_data.Y
# f = plt.figure()
# sp = f.add_subplot(111)
# X_kde = [item for sublist in np.ndarray.tolist(X) for item in sublist]
# orig_density = gaussian_kde(X_kde)
# x_domain = np.linspace(min(X_kde) - abs(max(X_kde)), max(X_kde)+abs(max(X_kde)), num_obs)
# sp.plot(x_domain, orig_density(x_domain))

