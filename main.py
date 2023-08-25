# Fabio: life cycle model of consumption, savings anf FLS
#        It also features a separate budget for pension benefits,
#        following the reform of the German pension system

#!!!Note: the model cannot match the decomposition of the effect on pension points (mech+behav)
#+ also the effect on earnings seems extremely hard to get. Perhaps adding minijob helps,
#if thos imply very low wages and little working hours
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

#Wages 1% hihgher than baseline in t=3 only 
pWt = co.setup();pWt.w[3,:]=1.01*pWt.w[3,:]
ModWt = sol.solveEulerEquation(pWt,model='baseline')

#Lowe in t=3 only to mimin a 1% increase in net wages 
pτt = co.setup();pτt.τ[3]=1.01*p.τ[3]-0.01#1-(1-p.τ[3])*1.01
Modτt = sol.solveEulerEquation(pτt,model='baseline')


#Compute the Frisch 
#pΩ
#ModΩ
########################################
# simulate the model
########################################

#Baseline
SB= sim.simNoUncer_interp(p,ModB,Tstart=0,Astart=p.startA,Pstart=np.ones(p.N)*p.startP)

#Pension reform
SP= sim.simNoUncer_interp(p,ModP,Tstart=3,Astart=SB['A'][3,:],Pstart=SB['p'][3,:])

#Wages 1% higher than baseline in t=3 only 
SWt= sim.simNoUncer_interp(pWt,ModWt,Tstart=2,Astart=SB['A'][2,:],Pstart=SB['p'][2,:])

# Lower taxes in t=3 only 
Sτt= sim.simNoUncer_interp(pτt,Modτt,Tstart=2,Astart=SB['A'][2,:],Pstart=SB['p'][2,:])



# ########################################
# # compute key elasticities
# ########################################


#Frisch elasticity: %change id(t)n h for an expected 1% increase in wage w in t=3
ϵf_Wt=(np.mean(SWt['h'][3,:])/np.mean(SB['h'][3,:])-1)*100
ϵf_Wti=np.mean(SWt['h'][3,:][SB['h'][3,:]>0.0]/SB['h'][3,:][SB['h'][3,:]>0.0])
print("The Simulated Frisch Elasticity (using change in w) is {}, intensive margin is {}".format(ϵf_Wt,ϵf_Wti))

ϵf_τt=(np.mean(Sτt['h'][3,:])/np.mean(SB['h'][3,:])-1)*100
ϵf_τti=np.mean(Sτt['h'][3,:][SB['h'][3,:]>0.0]/SB['h'][3,:][SB['h'][3,:]>0.0])
print("The Simulated Frisch Elasticity (using change in w) is {}, intensive margin is {}".format(ϵf_τt,ϵf_τti))


# ########################################
# # plot the result
# ########################################

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
plt.xlabel("Age")
plt.legend(('Reform','No Reform','Wages increase'))
plt.ylabel("Hours over time")
plt.show()




# 4
fig = plt.figure(figsize=(10,4)) 
plt.plot(t,np.mean(SP['A'],axis=1), 'blue', t, np.mean(SB['A'],axis=1), 'red')
plt.xlabel("Age")
plt.legend(('Reform','No Reform'))
plt.ylabel('Assets over time')
plt.show()



# 5
fig = plt.figure(figsize=(10,4)) 
plt.plot(t,np.mean(SP['p'],axis=1), 'blue', t, np.mean(SB['p'],axis=1), 'red')
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
ax.plot(p.agrid,ModB['V'][p.R-29,0,:,0,0], label="Value of FLP=0") 
ax.plot(p.agrid,ModB['V'][p.R-29,1,:,0,0], label="Value of FLP=1") 
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


print("WLP is {}, data is {}".format(np.mean(SB['h'][3:11,:]>0),0.64))
print("Part time is {}, data is {}".format(np.mean(SB['h'][3:11,:]==1),0.3136))
print("Effect on employment is {}, data is {}".format(np.mean(p.wls[SP['h'][3:11,:]]>0)-np.mean(p.wls[SB['h'][3:11,:]]>0),0.1))
print("The effect on full time employment is {}, data is {}".format(np.mean(SP['h'][3:11,:][SP['h'][3:11,:]>0]==2)-np.mean(SB['h'][3:11,:][SB['h'][3:11,:]>0]==2),0.045))
print("Increase in employment is {}, data is {}".format(np.mean(SP['h'][3:11,:]>0)-np.mean(SB['h'][3:11,:]>0),0.099))
print("Effect on earnings is {}, data is {}".format((np.mean(p.wls[SP['h'][3:11,:]]*SP['wh'][3:11,:])-np.mean(p.wls[SB['h'][3:11,:]]*SB['wh'][3:11,:]))/(np.mean(p.wls[SB['h'][3:11,:]]*SB['wh'][3:11,:])),0.46))
print("Baseline pension points are {}, data is {}".format(np.mean(np.diff(SB['p'][3:11,:],axis=0)),0.23))
print("Gender gap in old age if {}".format(1-(np.mean(p.ρ*SB['p'][p.R:,:]))/np.mean(p.y_N[p.R:,:])))



