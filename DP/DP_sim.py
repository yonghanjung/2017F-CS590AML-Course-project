import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import mixture
from scipy.stats import gaussian_kde
from Data_generation import Data_generaiton


class DP_Sim(object):
    np.random.seed(123)
    def __init__(self, num_cls, num_data):
        self.num_cls = num_cls
        self.num_data = num_data
        alpha = np.random.uniform(0,100,self.num_cls)
        self.weights = np.random.dirichlet(alpha,size=1)[0]

    def Data_generation(self):
        mus = np.random.normal(0, 100, self.num_cls)
        sigs = np.random.uniform(0.5, 10, self.num_cls)

        X = list()
        true_C = list()
        for idx in range(self.num_data):
            cls_idx = np.random.choice(list(range(0, self.num_cls)), 1, list(self.weights))
            true_C.append(cls_idx)
            mu = mus[cls_idx]
            sig = sigs[cls_idx]
            xi = np.random.normal(mu, sig, 1)[0]
            X.append(xi)
        return X
        # self.C = true_C
        # self.mus = mus
        # self.sigs = sigs
        # return X, true_C, mus, sigs

    def preprocess(self, X):
        if X.shape[1] >= X.shape[0] or pd.isnull(X.shape[1]):
            X = np.matrix(X).T
        else:
            pass
        return X

    def DP_model(self, X, init_compo = 20 ):
        X = np.reshape(X, (len(X), 1))
        X = self.preprocess(X)
        dpgmm = mixture.BayesianGaussianMixture(
            n_components=init_compo, weight_concentration_prior=1 / self.num_data,
            max_iter=1000, tol=1e-10,).fit(X)
        self.dpgmm = dpgmm

    def graphs(self, X):
        X = np.reshape(X, (len(X), 1))
        X = self.preprocess(X)

        f = plt.figure()

        orig_plot = f.add_subplot(211)
        X_kde = np.ndarray.flatten(X)
        # X_kde = [item for sublist in np.ndarray.tolist(X) for item in sublist]
        orig_density = gaussian_kde(X_kde)
        x_domain = np.linspace(min(X_kde) - abs(max(X_kde)), max(X_kde)+abs(max(X_kde)), self.num_data)
        orig_plot.plot(x_domain, orig_density(x_domain))
        orig_plot.hist(X_kde,100,normed=True)

        sim_plot = f.add_subplot(212)
        X_sim = self.dpgmm.sample(self.num_data)[0]
        X_sim = np.ndarray.flatten(X_sim)
        # X_sim= [item for sublist in X_sim for item in sublist]
        sim_density = gaussian_kde(X_sim)
        # sim_x_domain = np.linspace(min(X_sim) - 5, max(X_sim) + 5, self.num_data)
        sim_plot.plot(x_domain, sim_density(x_domain))
        sim_plot.hist(X_sim,100,normed=True)

        return f

# num_obs = 50000
# num_intv = 100
# dim = 5
# gen_data = Data_generaiton(num_obs, num_intv, dim)
# gen_data.data_gen()
# X = gen_data.Y_intv
#
# DP_sim = DP_Sim(10,num_obs)
# # X = DP_sim.Data_generation()
# DP_sim.DP_model(X,20)
# f = DP_sim.graphs(X)
#
