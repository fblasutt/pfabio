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
# pWt = co.setup();pWt.w[3,:]=1.01*pWt.w[3,:]
# ModWt = sol.solveEulerEquation(pWt,model='baseline')

# #Wages 1% hihgher than baseline for all t
# pWp = co.setup();pWp.w=1.01*pWp.w
# ModWp = sol.solveEulerEquation(pWp,model='baseline')

# #Lower permanent taxes such that post-tax wages are 1% higher than baseline  for all t
# pτp = co.setup();pτp.τ=1-1.01*(1-pτp.τ)
# Modτp = sol.solveEulerEquation(pτp,model='baseline')

#Compute the Frisch 
#pΩ
#ModΩ
########################################
# simulate the model
########################################

#Baseline
SB= sim.simNoUncer_interp(p,ModB,Tstart=0,Astart=np.ones(p.N)*p.startA,Pstart=np.zeros(p.N))

#Pension reform
SP= sim.simNoUncer_interp(p,ModP,Tstart=3,Astart=SB['A'][3,:],Pstart=SB['p'][3,:])

# #Wages 1% higher than baseline in t=3 only 
# SWt= sim.simNoUncer_interp(pWt,ModWt,Tstart=2,Astart=SB['A'][2,:],Pstart=SB['p'][2,:])

# #Wages 1% higher than baseline for all t 
# SWp= sim.simNoUncer_interp(pWp,ModWp,Tstart=3,Astart=SB['A'][3,:],Pstart=SB['p'][3,:])

# #Lower taxes such that post-tax wages are 1% higher than baseline  
# Sτp= sim.simNoUncer_interp(pτp,Modτp,Tstart=3,Astart=SB['A'][3,:],Pstart=SB['p'][3,:])

# #Wages 1% higher than baseline in t=3 only: UNEXPECTED CHANGE 
# SWt_un= sim.simNoUncer_interp(pWt,ModWt,Tstart=3,Astart=SB['A'][3,:],Pstart=SB['p'][3,:])

# #Wages are permanently higher but I control for wealth
# SWp_comp= sim.simNoUncer_interp(pWp,ModWp,Tstart=3,Astart=SB['A'][3,:],Pstart=SB['p'][3,:],Vstart=ModB['V'])

########################################
# compute key elasticities
########################################


# #Frisch elasticity: %change id(t)n h for an expected 1% increase in wage w in t=3
# ϵf_Wt=(np.mean(SWt['h'][3,:])/np.mean(SB['h'][3,:])-1)*100
# ϵf_Wti=np.mean(SWt['h'][3,:][SB['h'][3,:]>0.0]/SB['h'][3,:][SB['h'][3,:]>0.0])
# print("The Simulated Frisch Elasticity is {}, intensive margin is {}".format(ϵf_Wt,ϵf_Wti))



# #Hicks elasticity: %change in h for an unxpected 1% increase in wage w in t=3
# ϵh_Wp_comp=np.mean(SWp_comp['h'][3,:])/np.mean(SB['h'][3,:])
# print("The Simulated (life-cycle) Hicks Elasticity is {}".format(ϵh_W p_comp))

# #Marshallian elasticity: %change in h for an expected 1% increase in wage w forall t
# ϵm_Wp=np.mean(SWp['h'][3,:])/np.mean(SB['h'][3,:])
# ϵm_τ=np.mean(Sτp['h'][3,:])/np.mean(SB['h'][3,:])
# print("The Marshallian Elasticity is {}, computed using taxes is {}".format(ϵm_Wp,ϵm_τ))

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
plt.plot(t,np.mean(SP['h'],axis=1), 'blue',
         t, np.mean(SB['h'],axis=1), 'red')
        # t, np.mean(SWt['h'],axis=1), 'orange')
plt.xlabel("Age")
plt.legend(('Reform','No Reform','Wages increase'))
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

#Graph the value of participating or not in the labor market 2 periods before retirement
fig, ax = plt.subplots(figsize=(11, 8))   #Initialize figure and size
ax.plot(p.agrid,ModB['V'][p.R-2,0,:,0,0], label="Value of FLP=0") 
ax.plot(p.agrid,ModB['V'][p.R-2,1,:,0,0], label="Value of FLP=1") 
#ax.plot(p.agrid,ModB['V'][p.R-2,2,:,0,0], label="Value of FLP=0") 
#ax.plot(p.agrid,ModB['V'][p.R-2,1,:,0,0], label="Value of FLP=1") 
ax.grid()
ax.set_xlabel('Assets')                   #Label of x axis
ax.set_ylabel('Utility')                  #Label of y axis
plt.legend()                              #Plot the legend
plt.show()                                #Show the graph


#Graph difference in the value of participating and not in the mkt
fig, ax = plt.subplots(figsize=(11, 8))   #Initialize figure and size
ax.plot(p.agrid,ModB['V'][p.R-2,0,:,0,0]-ModB['V'][p.R-2,0,:,0,0], label="Value(FLP=0)-Value(FLP=1)") 
ax.grid()
ax.set_xlabel('Assets')                   #Label of x axis
ax.set_ylabel('Utility')                  #Label of y axis
plt.legend()                              #Plot the legend
plt.show()                                #Show the graph


#Consumption under participation and not participation
fig, ax = plt.subplots(figsize=(11, 8))   #Initialize figure and size
ax.plot(p.agrid[:],ModB['c'][p.R-2,1,:,0,0], label="Cons if FLP=1") 
ax.plot(p.agrid[:],ModB['c'][p.R-2,0,:,0,0], label="Cons if FLP=0") 
ax.grid()
ax.set_xlabel('Assets')                   #Label of x axis
ax.set_ylabel('Consumption')              #Label of y axis
plt.legend()                              #Plot the legend
plt.show()     

fig, ax = plt.subplots(figsize=(11, 8))   #Initialize figure and size
ax.plot(p.agrid[:],ModB['A'][p.R-2,1,:,0,0], label="Cons if FLP=1") 
ax.plot(p.agrid[:],ModB['A'][p.R-2,0,:,0,0], label="Cons if FLP=0") 
ax.grid()
ax.set_xlabel('Assets')                   #Label of x axis
ax.set_ylabel('Consumption')              #Label of y axis
plt.legend()                              #Plot the legend
plt.show()     


print("WLP is {}".format(np.mean(SB['h'][3:11,:]>0)))
print("Increase in employment is {}, data is {}".format(np.mean(SP['h'][3:11,:]>0)-np.mean(SB['p'][3:11,:]>0),0.11))
print("Increase in hours is {}, data is {}".format(np.mean(SP['h'][3:11,:]>0)/np.mean(SB['p'][3:11,:]>0)-1, 0.46))