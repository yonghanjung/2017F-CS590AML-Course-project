from cvxpy import *
import numpy as np

# problem data

x = Variable()
px = 0.3
objective = Minimize(x)
constraints = [ px*log(px) - px*log(x)  + (1-px)*log(1-px) - (1-px) * (1-log(1-x)) >= 0.0  ,
               0 <= x,
               x <= 1]
prob = Problem(objective, constraints)

result = prob.solve()
