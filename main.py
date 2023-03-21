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


# set up parameters
p = co.setup()



########################################
# solve the model
########################################

#Pension reform
ModP= sol.solveEulerEquation(p,model='pension reform')

#Baseline
ModB = sol.solveEulerEquation(p,model='baseline')

# #Wages 1% hihgher than baseline in t=3 only 
# pp = co.setup();pp.w[3,:]=1.01*pp.w[3,:]
# ModTt = sol.solveEulerEquation(pp,model='baseline')

# #Wages 1% hihgher than baseline for all t
# ppp = co.setup();ppp.w=1.01*ppp.w
# ModTp = sol.solveEulerEquation(ppp,model='baseline')

# #Lower taxes such that post-tax wages are 1% higher than baseline  for all t
# pppp = co.setup();pppp.τ=1-1.01*(1-pppp.τ)
# ModTτ = sol.solveEulerEquation(pppp,model='baseline')

# ########################################
# # simulate the model
# ########################################

#Baseline
SB= sim.simNoUncer_interp(p,ModB,Tstart=0,Astart=np.ones(p.N)*p.startA,Pstart=np.zeros(p.N))

#Pension reform
SP= sim.simNoUncer_interp(p,ModP,Tstart=3,Astart=SB['A'][3,:],Pstart=SB['p'][3,:])

# #Wages 1% higher than baseline in t=3 only 
# STt= sim.simNoUncer_interp(pp,ModTt,Tstart=0,Astart=np.ones(p.N)*p.startA,Pstart=np.zeros(p.N))

# #Wages 1% higher than baseline for all t 
# STp= sim.simNoUncer_interp(ppp,ModTp,Tstart=3,Astart=SB['A'][3,:],Pstart=SB['p'][3,:])

# #Lower taxes such that post-tax wages are 1% higher than baseline  
# STτ= sim.simNoUncer_interp(pppp,ModTτ,Tstart=3,Astart=SB['A'][3,:],Pstart=SB['p'][3,:])

# ########################################
# # compute key elasticities
# ########################################


# #Frisch elasticity: %change in h for an expected 1% increase in wage w in t=3
# ϵf=np.mean(np.diff(STt['h'][:,:],axis=0)[2])/np.mean(STt['h'][2,:])*100
# ϵf_t=1.0/p.γh*(p.maxHours-np.mean(STt['h'][2,:]))/np.mean(STt['h'][2,:])

# print("The Simulated Frisch Elasticity is {}, theoretical is {}".format(ϵf,ϵf_t,))

# #Marshallian elasticity: %change in h for an expected 1% increase in wage w forall t
# ϵm=(np.mean(STp['h'][3,:])/np.mean(SB['h'][3,:])-1)*100
# ϵm_τ=(np.mean(STτ['h'][3,:])/np.mean(SB['h'][3,:])-1)*100
# print("The Marshallian Elasticity is {}, computed using taxes is {}".format(ϵm,ϵm_τ))

########################################
# plot the result
########################################

t = list(range(30, p.T+30))
# Plot paths of consumption and assets 

# 1
fig = plt.figure(figsize=(10,4)) 
plt.plot(t,np.mean(SP['c'],axis=1), 'blue', t, np.mean(SB['c'],axis=1), 'red')
plt.xlabel("Age")
plt.legend(('Reform','No Reform'))
plt.ylabel("Consumption over time")
plt.show()


# 2
fig = plt.figure(figsize=(10,4)) 
plt.plot(t,np.mean(SP['wh'],axis=1), 'blue', t, np.mean(SB['wh'],axis=1), 'red')
plt.xlabel("Age")
plt.legend(('Reform','No Reform'))
plt.ylabel("Earnings over time")
plt.show()
    
    
# 3
fig = plt.figure(figsize=(10,4)) 
plt.plot(t,np.mean(SP['h'],axis=1), 'blue', t, np.mean(SB['h'],axis=1), 'red')
plt.xlabel("Age")
plt.legend(('Reform','No Reform'))
plt.ylabel("Hours over time")
plt.show()




# 4
fig = plt.figure(figsize=(10,4)) 
plt.plot(t,np.mean(SP['A'][1:p.T+1,:],axis=1), 'blue', t, np.mean(SB['A'][1:p.T+1,:],axis=1), 'red')
plt.xlabel("Age")
plt.legend(('Reform','No Reform'))
plt.ylabel('Assets over time')
plt.show()



# 5
fig = plt.figure(figsize=(10,4)) 
plt.plot(t,np.mean(SP['p'][1:p.T+1,:],axis=1), 'blue', t, np.mean(SB['p'][1:p.T+1,:],axis=1), 'red')
plt.xlabel("Age")
plt.legend(('Reform','No Reform'))
plt.ylabel('points')
plt.show()


# 6
fig = plt.figure(figsize=(10,4)) 
plt.plot(t,np.mean(SB['c'],axis=1), 'blue', t, np.mean(SB['wh'],axis=1), 'red')
plt.xlabel("Age")
plt.legend(('c','earn'))
plt.show()


print("Increase in pension points is {}".format(np.mean(np.diff(SP['p'],axis=0)[3:11,:])/np.mean(np.diff(SB['p'],axis=0)[3:11,:])-1))
print("WLP is {}".format(np.mean(SB['h'][3:11,:]>0)))