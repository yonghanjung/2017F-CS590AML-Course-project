import numpy as np
import scipy as sp
import pandas as pd
import sympy

def Simplex(CM,rH):
    '''
    Simplex algorithm (Maximization)
    :param CM: numpy matrix (coefficient matrix)
    :param rH: numpy array
    :return: numpy coefficient matrix after simplex
    '''

    # Initialization
    I,J = CM.shape
    for j in range(J):
        CM[0,j] = -CM[0,j]

    augmented_col = np.zeros((1, I)).T
    augmented_col[0,0] = 1
    CM = np.concatenate((augmented_col, CM), axis=1)
    I, J = CM.shape
    CM = np.concatenate((CM, rH), axis=1)

    # Run if any negative coefficient exists
    while( ( np.array(CM[0,])[0][1:J] <  0).any() ):
        # pick j*
        for j in range(J):
            if CM[0,j] < 0:
                j_star = j
                break
            else:
                continue

        # choose i*
        ratio = []
        for i in range(1,I):
            ratio_i = CM[i,J]/CM[i,j_star]
            ratio.append(ratio_i)
        i_star = np.argmin(ratio)+1
        CM_istar = CM[i_star,]

        # pivot
        for i in range(I):
            # skip i_star
            if i == i_star:
                CM[i,] = CM[i,] * 1/CM[i_star, j_star]
            else:
                pivot_ratio_i = - CM[i,j_star] / CM[i_star, j_star]
                pivot_row = CM_istar * pivot_ratio_i
                CM[i,] = CM[i,] + pivot_row

    CM_sol = CM[:,1:J]
    rH = CM[:,J]
    return CM_sol, rH



''' Main '''
# CM = np.matrix( [ [1,1,0,0],[2,1,1,0],[1,2,0,1] ]  )
# rH = np.matrix([0,4,3]).T
# CM_sol, rH_sol = Simplex(CM,rH)

CM = [[0,1,0,1,0,1,0,1], \
     [1,1,0,0,0,0,0,0], \
     [0,0,1,1,0,0,0,0], \
     [0,1,0,0,0,0,1,0], \
     [0,0,0,0,0,1,0,1]]
CM = np.matrix(CM)
rH = np.matrix( [0,0.1,0.2,0.3,0.4] ).T
CM_sol, rH_sol = Simplex(CM,rH)

