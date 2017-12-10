import numpy as np
import matplotlib.pyplot as plt

X = np.array(range(1,10000))/10000
p_obs = 0.069
output = p_obs*np.log(p_obs/X) + (1-p_obs)*np.log(p_obs/(1-X))