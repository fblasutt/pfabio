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
import sol_numba_parallel as sol
import sim
import matplotlib.pyplot as plt
import numpy as np

# set up parameters
par = co.setup()



########################################
# solve the model
########################################
reform = 1
[policyA1_1, policyh_1, policyC_1, V_1, policyp_1] = sol.solveEulerEquation(reform, par)
reform = 0
[policyA1_0, policyh_0, policyC_0, V_0, policyp_0] = sol.solveEulerEquation(reform, par)


########################################
# simulate the model
########################################
reform = 1
[cpath_1, apath_1, hpath_1, Epath_1, Epath_tau_1, vpath_1, EPpath_1, EPpath_c_1, EPpath_m_c_1, EPpath_behav_1, EPpath_behav_c_1, EPpath_behav_m_c_1 ] \
    = sim.simNoUncer_interp(reform, policyA1_1, policyC_1, policyh_1, V_1, par)
reform = 0
[cpath_0, apath_0, hpath_0, Epath_0, Epath_tau_0, vpath_0, EPpath_0, EPpath_c_0, EPpath_m_c_0, EPpath_behav_0, EPpath_behav_c_0, EPpath_behav_m_c_0 ] \
    = sim.simNoUncer_interp(reform, policyA1_0, policyC_0, policyh_0, V_0, par)
  

########################################
# plot the result
########################################

t = list(range(30, par.T+30))
# Plot paths of consumption and assets 

# 1
fig = plt.figure(figsize=(10,4)) 
plt.plot(t,cpath_1, 'blue', t, cpath_0, 'red')
plt.xlabel("Age")
plt.legend(('Reform','No Reform'))
plt.ylabel("Consumption over time")
plt.show()


# 2
fig = plt.figure(figsize=(10,4)) 
plt.plot(t,Epath_1, 'blue', t, Epath_0, 'red')
plt.xlabel("Age")
plt.legend(('Reform','No Reform'))
plt.ylabel("Earnings over time")
plt.show()
    
    
# 3
fig = plt.figure(figsize=(10,4)) 
plt.plot(t,hpath_1, 'blue', t, hpath_0, 'red')
plt.xlabel("Age")
plt.legend(('Reform','No Reform'))
plt.ylabel("Hours over time")
plt.show()


# 4
fig = plt.figure(figsize=(10,4)) 
plt.plot(t,Epath_tau_1, 'blue', t, Epath_tau_0, 'red')
plt.xlabel("Age")
plt.legend(('Reform','No Reform'))
plt.ylabel("Gross Earnings over time")
plt.show()



# 10
fig = plt.figure(figsize=(10,4)) 
plt.plot(t,apath_1[1:par.T+1], 'blue', t, apath_1[1:par.T+1], 'red')
plt.xlabel("Age")
plt.legend(('Reform','No Reform'))
plt.ylabel('Assets over time')
plt.show()


# 11
fig = plt.figure(figsize=(10,4)) 
plt.plot(t,vpath_1, 'blue', t, vpath_0, 'red')
plt.xlabel("Age")
plt.legend(('Reform','No Reform'))
plt.ylabel('Value Function over time')
plt.show()


fig = plt.figure(figsize=(10,4)) 
plt.plot(t,cpath_0, 'blue', t, Epath_0, 'red')
plt.xlabel("Age")
plt.legend(('c','earn'))
plt.show()