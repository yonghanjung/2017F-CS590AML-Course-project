import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from Causal_bound import Causal_bound
from Causal_bound import causal_bound_case1
from Data_generation import Data_generation

execfile('Bandit/UCB.py')
execfile('Bandit/KL-UCB.py')
execfile('Bandit/B-KL-UCB.py')

window_size = 100

Bandit_UCB['Cum_regret_roll'] = Bandit_UCB['Cum_regret'].rolling(window=window_size ).mean()
Bandit_UCB['Cum_regret_roll'][pd.isnull(Bandit_UCB['Cum_regret_roll'])] = Bandit_UCB['Cum_regret'][pd.isnull(Bandit_UCB['Cum_regret_roll'])]
Bandit_UCB['Prob_opt_roll'] = Bandit_UCB['Prob_opt'].rolling(window=window_size ).mean()
Bandit_UCB['Prob_opt_roll'][pd.isnull(Bandit_UCB['Prob_opt_roll'])] = Bandit_UCB['Prob_opt'][pd.isnull(Bandit_UCB['Prob_opt_roll'])]

Bandit_KL['Cum_regret_roll'] = Bandit_KL['Cum_regret'].rolling(window=window_size ).mean()
Bandit_KL['Cum_regret_roll'][pd.isnull(Bandit_KL['Cum_regret_roll'])] = Bandit_KL['Cum_regret'][pd.isnull(Bandit_KL['Cum_regret_roll'])]
Bandit_KL['Prob_opt_roll'] = Bandit_KL['Prob_opt'].rolling(window=window_size ).mean()
Bandit_KL['Prob_opt_roll'][pd.isnull(Bandit_KL['Prob_opt_roll'])] = Bandit_KL['Prob_opt'][pd.isnull(Bandit_KL['Prob_opt_roll'])]

Bandit_BKL['Cum_regret_roll'] = Bandit_BKL['Cum_regret'].rolling(window=window_size ).mean()
Bandit_BKL['Cum_regret_roll'][pd.isnull(Bandit_BKL['Cum_regret_roll'])] = Bandit_BKL['Cum_regret'][pd.isnull(Bandit_BKL['Cum_regret_roll'])]
Bandit_BKL['Prob_opt_roll'] = Bandit_BKL['Prob_opt'].rolling(window=window_size ).mean()
Bandit_BKL['Prob_opt_roll'][pd.isnull(Bandit_BKL['Prob_opt_roll'])] = Bandit_BKL['Prob_opt'][pd.isnull(Bandit_BKL['Prob_opt_roll'])]


f, ax = plt.subplots(2, sharex='all')
plt.title("KL-UCB")
ax[0].set_title('Cumulative regret',size=20)
ax[0].plot(Bandit_UCB['Cum_regret_roll'],label="UCB")
ax[0].plot(Bandit_KL['Cum_regret_roll'] ,label="KL-UCB")
ax[0].plot(Bandit_BKL['Cum_regret_roll'], label="B-KL-UCB")
ax[0].legend(fontsize=13)
ax[0].set_ylabel('Cumulative regret',fontsize=15)

ax[1].set_title('Probability of choosing opt arm', size=20)
ax[1].plot(Bandit_UCB['Prob_opt_roll'], label="UCB")
ax[1].plot(Bandit_KL['Prob_opt_roll'], label="KL-UCB")
ax[1].plot(Bandit_BKL['Prob_opt_roll'], label="B-KL-UCB")
ax[1].legend(fontsize=13)
ax[1].set_xlabel('Trials', fontsize = 15) # X label
ax[1].set_ylabel('Probability',fontsize=15)





############################################################################
############################################################################
execfile('Bandit/LinUCB.py')
execfile('Bandit/B-LinUCB.py')

window_size = 20

Bandit_Lin['Cum_regret_roll'] = Bandit_Lin['Cum_regret'].rolling(window=window_size ).mean()
Bandit_Lin['Cum_regret_roll'][pd.isnull(Bandit_Lin['Cum_regret_roll'])] = Bandit_Lin['Cum_regret'][pd.isnull(Bandit_Lin['Cum_regret_roll'])]
Bandit_Lin['Prob_opt_roll'] = Bandit_Lin['Prob_opt'].rolling(window=window_size ).mean()
Bandit_Lin['Prob_opt_roll'][pd.isnull(Bandit_Lin['Prob_opt_roll'])] = Bandit_Lin['Prob_opt'][pd.isnull(Bandit_Lin['Prob_opt_roll'])]

Bandit_BLin['Cum_regret_roll'] = Bandit_BLin['Cum_regret'].rolling(window=window_size ).mean()
Bandit_BLin['Cum_regret_roll'][pd.isnull(Bandit_BLin['Cum_regret_roll'])] = Bandit_BLin['Cum_regret'][pd.isnull(Bandit_BLin['Cum_regret_roll'])]
Bandit_BLin['Prob_opt_roll'] = Bandit_BLin['Prob_opt'].rolling(window=window_size ).mean()
Bandit_BLin['Prob_opt_roll'][pd.isnull(Bandit_BLin['Prob_opt_roll'])] = Bandit_BLin['Prob_opt'][pd.isnull(Bandit_BLin['Prob_opt_roll'])]

f, ax = plt.subplots(2, sharex='all')
ax[0].set_title('Cumulative regret',size=20)
ax[0].plot(Bandit_Lin['Cum_regret_roll'],label="LinUCB")
ax[0].plot(Bandit_BLin['Cum_regret_roll'] ,label="B-LinUCB")
ax[0].legend(fontsize=13)
ax[0].set_ylabel('Cumulative regret',fontsize=15)

ax[1].set_title('Probability of choosing opt arm', size=20)
ax[1].plot(Bandit_Lin['Prob_opt_roll'], label="LinUCB")
ax[1].plot(Bandit_BLin['Prob_opt_roll'], label="B-LinUCB")
ax[1].legend(fontsize=13)
ax[1].set_xlabel('Trials', fontsize = 15) # X label
ax[1].set_ylabel('Probability',fontsize=15)


#
#
#
#
# f, ax = plt.subplots(2, sharex='all')
#
# plt.title("KL-UCB")
# ax[0].set_title('Cumulative regret',size=20)
# ax[0].plot(Bandit_UCB['Cum_regret'],label="UCB")
# ax[0].plot(Bandit_KL['Cum_regret'] ,label="KL-UCB")
# ax[0].plot(Bandit_BKL['Cum_regret'], label="B-KL-UCB")
# ax[0].legend(fontsize=13)
# ax[0].set_ylabel('Cumulative regret',fontsize=15)
#
# ax[1].set_title('Probability of choosing opt arm', size=20)
# ax[1].plot(Bandit_UCB['Prob_opt'], label="UCB")
# ax[1].plot(Bandit_KL['Prob_opt'], label="KL-UCB")
# ax[1].plot(Bandit_BKL['Prob_opt'], label="B-KL-UCB")
# ax[1].legend(fontsize=13)
# ax[1].set_xlabel('Trials', fontsize = 15) # X label
# ax[1].set_ylabel('Cumulative regret',fontsize=15)