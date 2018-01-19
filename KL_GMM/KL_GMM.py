import numpy as np
import scipy as sp
import pandas as pd
from DP.DP_sim import DP_Sim

def KL_Gaussian(f_mean, f_std, g_mean, g_std): # KL(f || g)
    return np.log( g_std / f_std ) + \
           ( f_std**2 + (f_mean - g_mean)**2 ) / (2*g_std ** 2) - \
           0.5

def KL_GMM(f_weights, g_weights, f_means, g_means, f_stds, g_stds):
    sum_result = 0
    for k in range(len(f_weights)):
        pi_k = f_weights[k]
        f_mean_k = f_means[k]
        f_stds_k = f_stds[k]

        sum_numer = 0
        sum_deno = 0
        for kdot in range(len(f_weights)):
            pi_kdot = f_weights[kdot]
            f_mean_kdot = f_means[kdot]
            f_stds_kdot = f_stds[kdot]

            sum_numer += pi_kdot*\
                         np.exp( -KL_Gaussian(f_mean_k,f_stds_k,f_mean_kdot, f_stds_kdot) )

        for h in range(len(g_weights)):
            nu_h = g_weights[h]
            g_mean_h = g_means[h]
            g_stds_h = g_stds[h]

            sum_deno += nu_h*\
                         np.exp(  -KL_Gaussian(f_mean_k, f_stds_k, g_mean_h, g_stds_h)  )

        sum_result += pi_k * np.log(  sum_numer/sum_deno )
    return sum_result

# num_obs_f = 1000
# num_cls_f = 20
#
# num_obs_g = 1000
# num_cls_g = 5
#
# DP_sim_f = DP_Sim(num_cls_f,num_obs_f)
# DP_sim_g = DP_Sim(num_cls_g,num_obs_g)
#
# X_f = DP_sim_f.Data_generation()
# X_g = DP_sim_g.Data_generation()
#
# DP_sim_f.DP_model(X_f,50)
# DP_sim_g.DP_model(X_g,50)
#
# f_weights = np.round( DP_sim_f.dpgmm.weights_,3 )
# g_weights = np.round( DP_sim_g.dpgmm.weights_,3 )
# f_means = np.round( DP_sim_f.dpgmm.means_,3 )
# g_means = np.round( DP_sim_g.dpgmm.means_,3 )
# f_stds = np.ndarray.flatten( np.round( np.sqrt(DP_sim_f.dpgmm.covariances_),3 ) )
# g_stds = np.ndarray.flatten( np.round( np.sqrt(DP_sim_g.dpgmm.covariances_),3 ) )
#
# kl_gmm = KL_GMM(f_weights, g_weights, f_means, g_means, f_stds, g_stds)