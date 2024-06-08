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


# set up parameters
p = co.setup()



########################################
# solve the model
########################################

#Pension reform
ModP= sol.solveEulerEquation(p,model='pension reform')

#Baseline
ModB = sol.solveEulerEquation(p,model='baseline')


#Compute the Frisch 
#pΩ
#ModΩ
########################################
# simulate the model
########################################

#Baseline
SB= sim.simNoUncer_interp(p,ModB,Tstart=0,Astart=p.startA,Pstart=np.ones(p.N)*p.startP,izstart=p.tw)

#Pension reform
SP= sim.simNoUncer_interp(p,ModP,Tstart=8,Astart=SB['A'][8,:],Pstart=SB['pb3'][8,:],izstart=SB['iz'][8,:])



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
plt.plot(t,np.mean(SP['pb'],axis=1), 'blue', t, np.mean(SB['pb'],axis=1), 'red')
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
plt.plot(t,np.mean(SB['c'],axis=1), 'blue', t, np.mean(SB['pb'],axis=1), 'red')
plt.xlabel("Age")
plt.legend(('c','earn'))
plt.show()

#Graph the value of participating or not in the labor market 2 periods before retirement
fig, ax = plt.subplots(figsize=(11, 8))   #Initialize figure and size
ax.plot(p.agrid,ModB['V'][p.R-29,1,:,0,0,0], label="Value of FLP=0") 
ax.plot(p.agrid,ModB['V'][p.R-29,0,:,0,0,0], label="Value of FLP=1") 
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
ax.plot(p.agrid[:],ModB['V'][0,1,:,0,0,0], label="Cons if FLP=1") 
ax.plot(p.agrid[:],ModB['V'][0,1,:,0,0,0], label="Cons if FLP=0") 
ax.grid()
ax.set_xlabel('Assets')                   #Label of x axis
ax.set_ylabel('Consumption')              #Label of y axis
plt.legend()                              #Plot the legend
plt.show()     

fig, ax = plt.subplots(figsize=(11, 8))   #Initialize figure and size
ax.plot(p.agrid[:],ModB['c'][52,0,:,1,8,1], label="Cons if FLP=1") 
ax.plot(p.agrid[:],ModB['c'][52,0,:,1,8,1], label="Cons if FLP=0") 
ax.grid()
ax.set_xlabel('Assets')                   #Label of x axis
ax.set_ylabel('Consumption')              #Label of y axis
plt.legend()                              #Plot the legend
plt.show()     



print("% Full time is {}, data is {}".format(np.mean(SB['h'][7,:]>=3),0.1984))
print("% part time is {}, data is {}".format(np.mean(SB['h'][7,:]==2),0.1986))
print("% marginal  is {}, data is {}".format(np.mean(SB['h'][7,:]==1),0.256))
print("Baseline earnings is {}, data is 7,694".format(np.mean(SB['wh'][7,:])*p.scale))
print("Baseline earnings >0 (non marginal) is {}, data is 15819".format(np.mean(SB['wh'][7,:][SB['h'][7,:]>1])*p.scale))
print("Pension points are {}, data is {}".format(np.nanmean(np.diff(SB['p'][7:9,:],axis=0)),0.23))

print("% Pension points <0 are {}, data is {}".format(np.nanmean((np.diff(SB['p'][7:9,:],axis=0)<=0.001)),0.638))
print("% Pension points [0.0,0.33] are {}, data is {}".format(np.nanmean((np.diff(SB['p'][7:9,:],axis=0)>0.001) & (np.diff(SB['p'][7:9,:],axis=0)<0.33)),0.098))
print("% Pension points [0.33,0.66] are {}, data is {}".format(np.nanmean((np.diff(SB['p'][7:9,:],axis=0)>0.33) & (np.diff(SB['p'][7:9,:],axis=0)<0.66)),0.147))
print("% Pension points [0.66,1] are {}, data is {}".format(np.nanmean((np.diff(SB['p'][7:9,:],axis=0)>0.66) & (np.diff(SB['p'][7:9,:],axis=0)<1)),0.075))
print("% Pension points>1 are {}, data is {}".format(np.nanmean(np.diff(SB['p'][7:9,:],axis=0)>1),0.043))


p033 =((np.diff(SB['p'][7:9,:],axis=0))>0.00).flatten() & ((np.diff(SB['p'][7:9,:],axis=0))<=0.33).flatten()
p3366=((np.diff(SB['p'][7:9,:],axis=0))>0.33).flatten() & ((np.diff(SB['p'][7:9,:],axis=0))<=0.66).flatten() 
p661 =((np.diff(SB['p'][7:9,:],axis=0))>0.66).flatten() & ((np.diff(SB['p'][7:9,:],axis=0))<=1.00).flatten()

print("Additional average caregiver pension points for whole population at *baseline* are {}, data is {}".format(np.mean(SB['pb2'][7,:]),0.0595))
print("Additional max    caregiver pension points for whole population  at *baseline* are {}, data is {}".format(np.max(SB['pb2'][7,:]),0.33))
print("Additional average caregiver pension points in p (0.00,0.33]     at *baseline* are {}, data is {}".format(np.mean(SB['pb2'][7,:][p033]),0.095))
print("Additional average caregiver pension points in p (0.33,0.66]     at *baseline* are {}, data is {}".format(np.mean(SB['pb2'][7,:][p3366]),0.245))
print("Additional average caregiver pension points in p (0.66,1.00]     at *baseline* are {}, data is {}".format(np.mean(SB['pb2'][7,:][p661 ]),0.189))


print("-----------------------------------------------------------------------------------------------------------------------------------------------------------")
print("Effect on hours is {}, data is 3.565 hours".format(
   (np.mean(SP['h'][8:12,:]==1)*10.0+np.mean(SP['h'][8:12,:]==2)*20.0+np.mean(SP['h'][8:12,:]==3)*38.5)-\
   (np.mean(SB['h'][8:12,:]==1)*10.0+np.mean(SB['h'][8:12,:]==2)*20.0+np.mean(SB['h'][8:12,:]==3)*38.5)))
        
print("The effect on all employment is {}, data is {}".format(np.mean(SP['h'][8:12,:]>0)-np.mean(SB['h'][8:12,:]>0),0.099))
print("The effect on full time employment is {}, data is {}".format(np.mean(SP['h'][8:12,:][SP['h'][8:12,:]>0]==3)-np.mean(SB['h'][8:12,:][SB['h'][8:12,:]>0]==3),0.045))
print("The effect on marginal employment is {}, data is {}".format(np.mean(SP['h'][8:12,:][SP['h'][8:12,:]>0]==1)-np.mean(SB['h'][8:12,:][SB['h'][8:12,:]>0]==1),-0.11))
print("The effect on regual employment is {}, data is {}".format(np.mean(SP['h'][8:12,:][SP['h'][8:12,:]>0]>1)-np.mean(SB['h'][8:12,:][SB['h'][8:12,:]>0]>1), 0.105))
print("Effect of pension point is {}, data is 0.153".format(np.nanmean(np.diff(SP['p'][8:13,:],axis=0))-np.nanmean(np.diff(SB['p'][8:13,:],axis=0))))
print("Effect of behavioral pension points is {}, data is 0.102".format(np.nanmean(np.diff(SP['p'][8:13,:],axis=0))-np.nanmean(np.diff(SB['p'][8:13,:],axis=0))-np.mean(SP['pb'][8:12,:])))
print("Effect on non-marginal earnings is {}, data is {}".format((np.mean(SP['wh'][8:12,:])-np.mean(SB['wh'][8:12,:]))*p.scale,2809))
print("Effect on non-marginal earnings (>0) is {}, data is {}".format((np.mean(SP['wh'][8:12,:][SP['wh'][8:12,:]>0])-np.mean(SB['wh'][8:12,:][SB['wh'][8:12,:]>0]))*p.scale,1588))


print("-----------------------------------------------------------------------------------------------------------------------------------------------------------")
#Get heterogeneous effects
beloww=SB['w'][2,:]<np.median(SB['w'][2,:])
print("Effect of earinngs point for rich women is {}".format(np.mean(np.diff(SP['p'][8:13,:][:,~(beloww)],axis=0))-np.mean(np.diff(SB['p'][8:13,:][:,~(beloww)],axis=0))))
print("Effect of earinngs point for poor women is {}".format(np.mean(np.diff(SP['p'][8:13,:][:,(beloww)],axis=0)) -np.mean(np.diff(SB['p'][8:13,:][:,(beloww)],axis=0))))
print("Effect of behavioral earinngs points for rich women is {}".format(np.mean(np.diff(SP['p'][8:13,:][:,~(beloww)],axis=0))-np.mean(np.diff(SB['p'][8:13,:][:,~(beloww)],axis=0))-np.mean(SP['pb'][8:12,:][:,~(beloww)])))
print("Effect of behavioral earinngs points for poor women is {}".format(np.mean(np.diff(SP['p'][8:13,:][:,(beloww)],axis=0))-np.mean(np.diff(SB['p'][8:13,:][:,(beloww)],axis=0))-np.mean(SP['pb'][8:12,:][:,(beloww)])))


for i in range(p.nw):
    print("Share in mini-jobs or no job for wage {} is {}".format(p.w[7,3,i],np.mean(SB['h'][7,:][SB['w'][7,:]==p.w[7,3,i]]<1)))
for i in range(p.nw):
    print("Share LS or no job          for wage {} is {}".format(p.w[8,3,i],np.mean((SP['h'][8,:]>SB['h'][8,:])[(SB['h'][8,:]<=1) & (SB['w'][8,:]==p.w[8,3,i])])))
print("-----------------------------------------------------------------------------------------------------------------------------------------------------------") 
 
print("Share of women increasing LS is {}, their avg points at basline are {}".format(np.mean(SP['h'][8:12,:]>SB['h'][8:12,:]) ,np.mean(np.diff(SB['p'][8:13],axis=0)[SP['h'][8:12,:]>SB['h'][8:12,:]]))) 
print("Share of women ==         LS is {}, their avg points at basline are {}".format(np.mean(SP['h'][8:12,:]==SB['h'][8:12,:]),np.mean(np.diff(SB['p'][8:13],axis=0)[SP['h'][8:12,:]==SB['h'][8:12,:]]))) 
print("Share of women decreasing LS is {}, their avg points at basline are {}".format(np.mean(SP['h'][8:12,:]<SB['h'][8:12,:]) ,np.mean(np.diff(SB['p'][8:13],axis=0)[SP['h'][8:12,:]<SB['h'][8:12,:]]))) 

