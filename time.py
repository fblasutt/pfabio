# Fabio: life cycle model of consumption, savings anf FLS
#        It also features a separate budget for pension benefits,
#        following the reform of the German pension system


######################################
# Preamble
######################################

# clear workspace and console
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass


# Import package
import co # user defined functions
import sol
import sim
import numpy as np
import matplotlib.pyplot as plt
from transitions_chart import transitions_chart
import time

# set up parameters
p = co.setup()



########################################
# solve the model
########################################


#Baseline
ModB = sol.solveEulerEquation(p,model='baseline')


#Baseline
SB= sim.simNoUncer_interp(p,ModB,Tstart=0,Astart=p.startA,Pstart=np.ones(p.N)*p.startP)


tic=time.time();ModB = sol.solveEulerEquation(p,model='baseline');toc=time.time()
print('Time elapsed for model solution is {}'.format(toc-tic))


tic=time.time();SB= sim.simNoUncer_interp(p,ModB,Tstart=0,Astart=p.startA,Pstart=np.ones(p.N)*p.startP);toc=time.time()
print('Time elapsed for model simulation is {}'.format(toc-tic))


plt.plot(SB['c'].mean(axis=1))