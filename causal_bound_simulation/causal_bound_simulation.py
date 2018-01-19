import numpy as np
import scipy as sp
import pandas as pd
from DP.DP_sim import DP_Sim
from KL_GMM.KL_GMM import KL_GMM
from Data_generation import Data_generaiton

num_obs = 10000
num_intv = 200
num_cls = 2 # meaningless
dim = 10

gen_data = Data_generaiton(num_obs, num_intv, dim)
gen_data.data_gen()

Obs = gen_data.Obs
Intv = gen_data.Intv

x = 0
Y = Obs[Obs['X'] == x]['Y'] # P(Y|x)
Y_intv = Intv[Intv['X']==x]['Y'] # P(Y|do(x))

# Y = gen_data.Y
# Y_intv = gen_data.Y_intv
#
DP_sim_obs = DP_Sim(num_cls,num_obs) # num_cls
DP_sim_intv = DP_Sim(num_cls,num_obs) # num_cls
dpgmm_obs = DP_sim_obs.DP_model(Y,50)
dpgmm_intv = DP_sim_intv.DP_model(Y_intv,50)
f_obs = DP_sim_obs.graphs(Y)
f_intv = DP_sim_intv.graphs(Y_intv)

Px = len(Obs[Obs['X'] == x]) / num_obs
M = -np.log(Px)


f_weights = np.round( DP_sim_obs.dpgmm.weights_,20 )
g_weights = np.round( DP_sim_intv.dpgmm.weights_,20 )
f_means = np.round( DP_sim_obs.dpgmm.means_,20 )
g_means = np.round( DP_sim_intv.dpgmm.means_,20 )
f_stds = np.ndarray.flatten( np.round( np.sqrt(DP_sim_obs.dpgmm.covariances_),3 ) )
g_stds = np.ndarray.flatten( np.round( np.sqrt(DP_sim_intv.dpgmm.covariances_),3 ) )

kl_gmm = KL_GMM(f_weights, g_weights, f_means, g_means, f_stds, g_stds)