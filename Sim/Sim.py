import numpy as np
import scipy.special as sp
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import mixture
from scipy.stats import gaussian_kde
from scipy.optimize import minimize

class MyError(Exception):
    pass

class DataGen(object):
    '''
    Class DataGen contains all data and methods related to
    generating data.
    - Input: D (dim), N (num_obs), Ns (num_sample_intv), Mode (easy, crazy)
    - Output: Observational data and Interventional data
    '''

    # Seed fix
    # Some good seed are 1234, 12345.
    np.random.seed(12345)
    def __init__(self,D,N,Ns,Mode):
        '''
        Initializing the class
        :param D: Dimension
        :param N: Number of observations and interventions (true)
        :param Ns: Number of sampled observation, Ns << N.
        :param Mode:
            - Easy: easy parametrization (Normal), assuming SCM is known
            - Crazy: crazy parametrization, arbitrary density
        '''

        '''
        Declaring input as global in the class
        to contain the input value as data. 
        '''
        self.dim = D
        self.num_obs = N
        self.num_intv = Ns
        self.Mode = Mode # easy / crazy

    def intfy(self,W):
        '''
        Transform the real-number array (e.g., 1.0,2.0) to 
        the integer array (e.g., 1, 2).  
        :param W: Real-number array 
        :return: Integer array
        '''
        return np.array(list(map(int, W)))

    def weird_projection(self, X):
        '''
        Project
        :param X:
        :return:
        '''
        X_proj = np.log(np.abs(X)+20) * np.arctan(X) + (10*np.sin(X)) - (np.tan(X ** 2) + 20) * np.exp(X-20)
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

        if self.Mode == 'crazy':
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
        if self.Mode == 'easy':
            Z = self.U1 + self.U2
        elif self.Mode == 'crazy':
            Z = np.exp(self.U1) + np.exp(self.U2)
        self.Z = ((Z - np.mean(Z, axis=0)) / np.var(Z))

    def gen_X(self):
        coef_xz = np.reshape(-2 * np.random.rand(self.dim), (self.dim, 1))
        coef_u1x = np.reshape(2 * np.random.rand(self.dim), (self.dim, 1))
        coef_u3x = np.reshape(-2 * np.random.rand(self.dim), (self.dim, 1))

        U1 = np.matrix(self.U1)
        U3 = np.matrix(self.U3)
        Z = np.matrix(self.Z)

        if self.Mode == 'easy':
            self.X_obs = self.intfy(  np.round( sp.expit(Z*coef_xz + U3 * coef_u3x + U1 * coef_u1x), 0) )
        elif self.Mode == 'crazy':
            X = np.exp(U1 * coef_u1x) - (U3 * coef_u3x) + np.abs(Z * coef_xz)
            X = np.round(  ((X - np.mean(X, axis=0)) / np.var(X)), 0)

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

        if self.Mode == 'easy':
            Y = U2 * coef_u2y + U3 * coef_u3y + Z * coef_zy + np.array(X_obs.T)
            Y_intv = U2 * coef_u2y + U3 * coef_u3y + Z * coef_zy + np.array(X_intv.T)
        elif self.Mode == 'crazy':
            Y = np.array(np.sin(U2 * coef_u2y)) * \
                np.array(-1 * np.array(-50 * np.tanh(U3 * coef_u3y)) +
                     (np.array(X_obs.T))) * \
                np.array(np.abs(Z * coef_zy + 1))
            Y_intv = np.array(np.sin(U2 * coef_u2y)) * \
                     np.array(-1 * np.array(-70 * np.tanh(U3 * coef_u3y)) +
                              (np.array(X_intv.T))) * \
                     np.array(np.abs(Z * coef_zy + 1))
        self.Y = 100 * ((Y - np.mean(Y, axis=0)) / np.var(Y))
        self.Y_intv = 100 * ((Y_intv - np.mean(Y, axis=0)) / np.var(Y))

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

        X_intv = np.asarray(X_intv)
        Y_intv = np.asarray(Y_intv)
        Intv_X = pd.DataFrame(X_intv)
        Intv_Y = pd.DataFrame(Y_intv)
        Intv_Z = pd.DataFrame(Z_obs)

        Intv_XY = pd.concat([Intv_X, Intv_Y], axis=1)
        Intv_XY.columns = ['X', 'Y']
        Intv_Z.columns = range(self.dim)

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

        self.Obs = pd.concat([Obs_XY, Obs_Z], axis=1)
        self.Intv = pd.concat([Intv_XY, Intv_Z], axis=1)
        self.Intv_S = pd.concat([SIntv_XY, SIntv_Z], axis=1)

    def data_gen(self):
        self.gen_U()
        self.gen_Z()
        self.gen_X()
        self.gen_Y()
        self.structure_data()

class DP_sim(DataGen):
    def __init__(self,D,N,Ns,Mode,x):
        self.x = x
        super().__init__(D,N,Ns,Mode)

        self.data_gen()
        self.Y_x = self.Obs[self.Obs['X']==self.x]['Y']
        self.Yinv_x = self.Intv[self.Intv['X']==self.x]['Y']
        self.Ysinv_x = self.Intv_S[self.Intv_S['X']==self.x]['Y']

    def preprocess(self, X):
        if X.shape[1] >= X.shape[0] or pd.isnull(X.shape[1]):
            X = np.matrix(X).T
        else:
            pass
        return X

    def Fit(self, X):
        X = np.reshape(X, (len(X), 1))
        X = self.preprocess(X)
        init_compo = 50
        dpgmm= mixture.BayesianGaussianMixture(
            n_components=init_compo, weight_concentration_prior=1 / self.num_obs,
            max_iter=1000, tol=1e-10, ).fit(X)
        return dpgmm

    def DP_fit(self):
        self.dpobs = self.Fit(self.Y_x)
        self.dpintv = self.Fit(self.Y_intv)
        self.dpintv_s = self.Fit(self.Ysinv_x)

class Graph(DP_sim):
    def Graph_obs_int(self):
        f = plt.figure()

        X_obs = np.reshape(self.Y_x, (len(self.Y_x), 1))
        X_obs = self.preprocess(X_obs)
        obs_plot = f.add_subplot(211)
        X_obs_kde = np.ndarray.flatten(X_obs)
        obs_density = gaussian_kde(X_obs_kde )
        x_domain_obs = np.linspace(min(X_obs_kde ) - abs(max(X_obs_kde )), max(X_obs_kde ) + abs(max(X_obs_kde)), self.num_obs)
        obs_plot.plot(x_domain_obs, obs_density(x_domain_obs))
        obs_plot.hist(X_obs, 100, normed=True)

        X_int = np.reshape(self.Yinv_x, (len(self.Yinv_x), 1))
        X_int = self.preprocess(X_int)
        int_plot = f.add_subplot(212)
        X_int_kde = np.ndarray.flatten(X_int)
        intv_density = gaussian_kde(X_int_kde)
        x_domain_int = x_domain_obs
        int_plot.plot(x_domain_int, intv_density(x_domain_int))
        int_plot.hist(X_int, 100, normed=True)
        return f

    def Graph_DPFit(self):
        if self.Mode == 'easy':
            print("Graph_DPFit method is only for 'crazy' mode")
            raise MyError

        f = plt.figure()

        X_int = np.reshape(self.Yinv_x, (len(self.Yinv_x), 1))
        X_int = self.preprocess(X_int)
        int_plot = f.add_subplot(211)
        X_int_kde = np.ndarray.flatten(X_int)
        intv_density = gaussian_kde(X_int_kde)
        x_domain_int = np.linspace(min(X_int_kde) - abs(max(X_int_kde)), max(X_int_kde) + abs(max(X_int_kde)),
                                   self.num_obs)
        int_plot.plot(x_domain_int, intv_density(x_domain_int))
        int_plot.hist(X_int, 100, normed=True)

        sim_plot = f.add_subplot(212)
        X_sim = self.dpintv.sample(self.num_obs)[0]
        X_sim = np.ndarray.flatten(X_sim)
        # X_sim= [item for sublist in X_sim for item in sublist]
        sim_density = gaussian_kde(X_sim)
        # sim_x_domain = np.linspace(min(X_sim) - 5, max(X_sim) + 5, self.num_data)
        sim_plot.plot(x_domain_int, sim_density(x_domain_int))
        sim_plot.hist(X_sim, self.num_obs/100, normed=True)
        return f

class KL_computation():
    def KL_Gaussian(self, f_mean, f_std, g_mean, g_std):  # KL(f || g)
        return np.log(g_std / f_std) + \
               (f_std ** 2 + (f_mean - g_mean) ** 2) / (2 * g_std ** 2) - \
               0.5

    def KL_GMM(self, f_weights, g_weights, f_means, g_means, f_stds, g_stds):
        sum_result = 0
        for k in range(len(f_weights)):
            reducing = 1e-6
            pi_k = f_weights[k]
            f_mean_k = f_means[k]
            f_stds_k = f_stds[k]
            if pi_k <= reducing:
                continue

            sum_numer = 0
            sum_deno = 0
            for kdot in range(len(f_weights)):
                pi_kdot = f_weights[kdot]
                if pi_kdot <= reducing:
                    continue
                f_mean_kdot = f_means[kdot]
                f_stds_kdot = f_stds[kdot]

                sum_numer += pi_kdot * \
                             np.exp(-self.KL_Gaussian(f_mean_k, f_stds_k, f_mean_kdot, f_stds_kdot))

            for h in range(len(g_weights)):
                nu_h = g_weights[h]
                if nu_h <= reducing:
                    continue
                g_mean_h = g_means[h]
                g_stds_h = g_stds[h]

                sum_deno += nu_h * \
                            np.exp(-self.KL_Gaussian(f_mean_k, f_stds_k, g_mean_h, g_stds_h))

            sum_result += pi_k * np.log( (sum_numer + reducing)  / (sum_deno + reducing))
        return sum_result


class CausalBound(DP_sim, KL_computation):
    def Set_x(self, x):
        self.x = x
        self.Y_x = self.Obs[self.Obs['X'] == self.x]['Y']
        self.Yinv_x = self.Intv[self.Intv['X'] == self.x]['Y']
        self.Ysinv_x = self.Intv_S[self.Intv_S['X'] == self.x]['Y']
        if self.Mode == 'crazy':
            self.DP_fit()
        # self.KL()
        # self.ComputeBound()

    def Entropy_x(self):
        px = len(self.Obs[ self.Obs['X']==self.x ])/self.num_obs
        return -np.log(px)


    def ComputeBound(self):
        minus_logpx = self.Entropy_x()
        if self.Mode == 'easy':
            true_mu_do = np.mean(self.Yinv_x)
            mu_obs = np.mean(self.Y_x)
            std_obs = np.std(self.Y_x)
            std_do = np.std(self.Ysinv_x)

            M = ( ( minus_logpx + 0.5 - np.log( std_do/std_obs) ) * 2*(std_do**2) ) - std_obs**2
            upper = mu_obs + np.sqrt(M)
            lower = mu_obs - np.sqrt(M)
            return [lower, true_mu_do, upper]

        elif self.Mode == 'crazy':
            self.DP_fit()
            Hx = self.Entropy_x()
            rounding_digit = 4
            f_means = np.round(self.dpobs.means_, rounding_digit)
            true_means = np.round(self.dpintv.means_, rounding_digit)
            f_weights = np.round(self.dpobs.weights_, rounding_digit)
            g_weights = np.round(self.dpintv_s.weights_, rounding_digit)
            f_stds = np.ndarray.flatten(np.round(np.sqrt(self.dpobs.covariances_), rounding_digit))
            g_stds = np.ndarray.flatten(np.round(np.sqrt(self.dpintv_s.covariances_), rounding_digit))

            cons = ({'type': 'ineq',
                     'fun': lambda x: Hx - self.KL_GMM(f_weights, g_weights, f_means, x, f_stds, g_stds)[0]},
                    {'type': 'ineq',
                     'fun': lambda x: self.KL_GMM(f_weights, g_weights, f_means, x, f_stds, g_stds)[0]}
                    )

            x0 = f_means

            min_fun = lambda mu_do: np.sum(mu_do * g_weights)
            max_fun = lambda mu_do: -np.sum(mu_do * g_weights)

            max_iter = 1000
            lower = minimize(min_fun, x0=x0, constraints=cons, method='SLSQP',options={'maxiter':max_iter,'disp':True})
            upper = minimize(max_fun, x0=x0, constraints=cons, method='SLSQP',options={'maxiter':max_iter,'disp':True})
            # return [np.sum(lower.x * g_weights),np.sum(true_means * g_weights),np.sum(upper.x*g_weights)], lower, upper
            return [np.sum(lower.x * g_weights), np.sum(true_means * g_weights),
                    np.sum(upper.x * g_weights)]

class B_KLUCB(CausalBound):
    '''
    Currently this class is only applicable for 'easy' mode
    '''
    def __init__(self, D,N,Ns,Mode,T, K):
        super().__init__(D, N, Ns, Mode, x)
        if self.Mode == 'crazy':
            print("Crazy mode is not allowed")
            raise MyError()
        self.T = T # number of iteration of bandits
        self.K = K # number of arms

    def CutArm(self):
        # arm collection
        ## arm collection for arm 0
        self.Set_x(x=0)
        bd_0 = self.ComputeBound()
        lb0 = bd_0[0]
        ub0 = bd_0[2]
        arm_0 = [lb0, ub0]

        ## arm collection for arm 1
        self.Set_x(x=1)
        bd_1 = self.ComputeBound()
        lb1 = bd_1[0]
        ub1 = bd_1[2]
        arm_1 = [lb1, ub1]
        arms = [arm_0, arm_1]

        ## arm collection for lb and ub for each arms.
        l_arm = [lb0, lb1]
        u_arm = [ub0, ub1]
        l_max = np.max(l_arm)

        # Cutting arm
        arms_new = dict()
        l_arm_new = dict()
        u_arm_new = dict()
        arms_idx = list()

        for a in range(K):
            # cur arm condition
            if u_arm[a] < l_max:
                continue
            else:
                arms_new[a] = arms[a]
                l_arm_new[a] = l_arm[a]
                u_arm_new[a] = u_arm[a]
                arms_idx.append(a)

        # Declare remaing arms
        self.arms = arms_new
        self.l_arm = l_arm_new
        self.u_arm = u_arm_new
        self.armidx = arms_idx
        self.K = len(self.armidx)

    def Receive_rewards(self,arm,num_armpull):
        pulling_idx = num_armpull[arm]-1
        return list(self.Intv[self.Intv['X']==arm]['Y'])[pulling_idx]


    def Bandit_Init(self):
        '''
        Pulling each arm one time.
        :return:
        '''

        Arm = []
        Reward = []

        num_armpull = dict()
        reward_armpull = dict()
        for a in self.armidx:
            # initial arm pull
            num_armpull[a] = 1
            reward_armpull[a] = [self.Receive_rewards(a,num_armpull)]
        




    def Bandit(self):
        self.CutArm()
        if len(self.armidx) == 1:
            return self.armidx

        else:
            self.Bandit_Init()







#################### MAIN ############################
D = 3
N = 1000
Ns = 500
Mode = 'easy'
x = 0
T = 1000
K = 2
bkl = B_KLUCB(D,N,Ns,Mode,T,K)
graph = Graph(D,N,Ns,Mode,x)
# result, lower, upper = cb.ComputeBound()
# cb.Graph_obs_int()
# cb.Graph_DPFit()





