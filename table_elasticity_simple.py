# Import package
import co # user defined functions
import sol
import sim
import numpy as np


# set up parameters
p = co.setup()

beg=3
end=11

#######################################################################
# compute the baseline and obtain the "increase" in net wages that implies
# the same increse in income of the pension caregiver credit reform *if behavior
# is left constant*
#######################################################################

adjust=np.ones((p.T,p.N))/((1+p.r)**(np.cumsum(np.ones(p.T))-1.0))[:,None]



increase=0.083
p.tax[beg:end] = -increase;p.wls_point=np.array([0.0,0.0,1.0,1.0]);p.wls_point2=np.array([0.0,0.0,1.0,1.0]);ModB = sol.solveEulerEquation(p,model='baseline')
SB= sim.simNoUncer_interp(p,ModB,Tstart=np.zeros(p.N,dtype=np.int16),Astart=p.startA,Pstart=np.ones((p.T,p.N))*p.startP,izstart=p.tw)

p1 = co.setup()
p1.wls_point=np.array([0.0,0.0,1.0,1.0]);p1.wls_point2=np.array([0.0,0.5,1.5,1.5]);ModB1 = sol.solveEulerEquation(p1,model='baseline')
SB1= sim.simNoUncer_interp(p1,ModB1,Tstart=np.zeros(p.N,dtype=np.int16),Astart=p1.startA,Pstart=np.ones((p1.T,p1.N))*p1.startP,izstart=p1.tw)


income_pre_retirement =np.sum(adjust*SB['income_mod'])#how much more pre ret income in 1)?
income_post_retirement=np.sum(adjust*SB1['income_mod'])#avg income you get per pension point


print("Income pre-post retirement is {}".format(income_pre_retirement-income_post_retirement))
 
######################################################################"
# Below:
# 1) The actual caregiver credits reform without upper threshold
# 2) A reform with an increase in gross wages equivalent to "increase"
# 3) Compute elasticities associated with 1) and 2)
########################################################################"

#1) Pension caregiver reform
pρ = co.setup();pρ.wls_point2=np.array([0.0,0.5,1.5,1.5]);pρ.standard_wls=False#point_equivalent
Modρ = sol.solveEulerEquation(pρ,model='baseline')
Sρ= sim.simNoUncer_interp(pρ,Modρ,Tstart=np.zeros(p.N,dtype=np.int16),Astart=p.startA,Pstart=np.ones((p.T,p.N))*p.startP,izstart=p.tw)

#2) Increase in gross wages
pτ = co.setup();pτ.w[beg:end]=(1.0+increase)*p.w[beg:end]
Modτ = sol.solveEulerEquation(pτ,model='baseline')
Sτ= sim.simNoUncer_interp(pτ,Modτ,Tstart=np.zeros(p.N,dtype=np.int16),Astart=p.startA,Pstart=np.ones((p.T,p.N))*p.startP,izstart=p.tw)


#3) Compute elasticities associated with 1) and 2)
ϵρ = (co.hours_value(pρ,Sρ,beg,end).mean()/co.hours_value(p,SB,beg,end).mean()-1)/increase
ϵτ = (co.hours_value(pτ,Sτ,beg,end).mean()/co.hours_value(p,SB,beg,end).mean()-1)/increase



print("The forward elasticity is {}, the standard elasticity is {}".format(ϵρ,ϵτ))


# ######################################################################"
# # Below, compute marshallian, hicksian and firsh elasticities 
# # 1) Marhsall
# # 2) Hicks
# # 3) Frisch
# # 4) Display 1),2) and 3)
# ########################################################################


# increase_small =increase#with 0.001 lower increase

# ϵM=list();ϵH=list();ϵF=list();ϵMS=list()

# for i in range(beg,end):
    
#     # 1) Marhsall: perturb the whole profile of earnings
#     pM = co.setup();pM.w[i:]=(1.0+increase_small)*p.w[i:]
#     ModM = sol.solveEulerEquation(pM,model='baseline')
#     SM=sim.simNoUncer_interp(pM,ModM,Tstart=np.zeros(p.N,dtype=np.int16)+i,Astart=SB['A'],Pstart=SB['p'],izstart=SB['iz'])
    
#     # 2) Hicks elasticity
#     pH = co.setup();pH.w[i]=(1.0+increase_small)*p.w[i]
#     ModH = sol.solveEulerEquation(pH,model='baseline')
#     SH=sim.simNoUncer_interp(pH,ModH,Tstart=np.zeros(p.N,dtype=np.int16)+i,Astart=SB['A'],Pstart=SB['p'],izstart=SB['iz'])
    
#     # 3) Frisch elasticity
#     pF = co.setup();pF.w[i]=(1.0+increase_small)*p.w[i]
#     ModF = sol.solveEulerEquation(pF,model='baseline')
#     SF=sim.simNoUncer_interp(pF,ModF,Tstart=np.zeros(p.N,dtype=np.int16),Astart=p.startA,Pstart=np.ones((p.T,p.N))*p.startP,izstart=p.tw)

#     # 1*) Marshall forward elasticity
#     reti_income = np.sum((SB['ir'][i:]==0)*SB['income_mod'][i:]*adjust[i:],axis=0).mean()#how much more pre ret income in 1)?
#     extra_point=np.sum((SB['ir'][i:]==0)*SB['pb'][i:],axis=0).mean()#avg accumulated pension points during life
#     income_per_point=np.sum(extra_point*(SB['ir'][i:]==1)*adjust[i,:]*SB['income_mod'][i,:]/SB['p'][i,:],axis=0).mean()#avg income you get per pension point
    
#     reti_income = np.sum((SB['ir'][i:]==0)*SB['income_mod'][i:]*adjust[i:],axis=0).mean()#how much more pre ret income in 1)?
#     extra_point=np.sum(SB['pb'][i:],axis=0)#avg accumulated pension points during life
#     income_per_point=(extra_point*np.sum((SB['ir']==1)*adjust*SB['income_mod']/SB['p'],axis=0)).mean()#avg income you get per pension point


#     increasep=reti_income*increase_small/(income_per_point)
    
#     pMS = co.setup();pMS.points_base = 1.0+increasep
#     ModMS = sol.solveEulerEquation(pMS,model='baseline')
#     SMS=sim.simNoUncer_interp(pMS,ModMS,Tstart=np.zeros(p.N,dtype=np.int16)+i,Astart=SB['A'],Pstart=SB['p'],izstart=SB['iz'])


#     #4) Compute elasticities associated with 1),2) and 3)
#     ϵM.append((co.hours_value(p,SM,i,i+1).mean()/co.hours_value(p,SB,i,i+1).mean()-1)/increase_small)
#     ϵH.append((co.hours_value(p,SH,i,i+1).mean()/co.hours_value(p,SB,i,i+1).mean()-1)/increase_small) 
#     ϵF.append((co.hours_value(p,SF,i,i+1).mean()/co.hours_value(p,SB,i,i+1).mean()-1)/increase_small) 
#     ϵMS.append((co.hours_value(pMS,SMS,i,i+1).mean()/co.hours_value(p,SB,i,i+1).mean()-1)/increase_small) 

# print("Standard: The Marshallian elasticity is {}, Hicks is {}, Frisch is {}".format(np.array(ϵM).mean(),np.array(ϵH).mean(),np.array(ϵF).mean()))
# print("FForward: The Marshallian elasticity is {}                           ".format(np.array(ϵMS).mean()))


# # 1) Marhsall: perturb the whole profile of earnings 
# pMo = co.setup();pMo.w[beg:]=(1.0+increase_small)*p.w[beg:] 
# ModMo = sol.solveEulerEquation(pMo,model='baseline') 
# SMo= sim.simNoUncer_interp(pMo,ModMo,Tstart=np.zeros(p.N,dtype=np.int16)+beg,Astart=SB['A'],Pstart=SB['p'],izstart=SB['iz']) 
 
 




