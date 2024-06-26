# Import package
import co # user defined functions
import sol
import sim
import numpy as np


# set up parameters
p = co.setup()

beg=8
end=12

#######################################################################
# compute the baseline and obtain the "increase" in net wages that implies
# the same increse in income of the pension caregiver credit reform *if behavior
# is left constant*
#######################################################################

#baseline
increase=0.0701
p.tax[beg:end] = -increase;ModB = sol.solveEulerEquation(p,model='baseline')
SB= sim.simNoUncer_interp(p,ModB,Astart=p.startA,Pstart=np.ones(p.N)*p.startP,izstart=p.tw)
adjust=np.ones(SB['c'].shape)/((1+p.r)**(np.cumsum(np.ones(p.T))-1.0))[:,None]


#Higher pension points at constant behavior
pρ0 = co.setup();pρ0.Pmax=10000.145;pρ0.add_points_exp=1.5;pρ0.add_points=1.0#point_equivalent
beg=beg
Modρ0 = sol.solveEulerEquation(pρ0,model='pension reform')
Sρ0= sim.simNoUncer_interp(pρ0,Modρ0,Tstart=beg,Astart=SB['A'][beg,:],Pstart=SB['p'][beg,:],izstart=SB['iz'][beg,:])


#ratio below should be 1
print("ratio is {}".format((adjust[beg:]*Sρ0['income_mod'][beg:]).mean()/(adjust[beg:]*SB['income_mod'][beg:]).mean()))


######################################################################"
# Below:
# 1) The actual caregiver credits reform without upper threshold
# 2) A reform with an increase in gross wages equivalent to "increase"
# 3) Compute elasticities associated with 1) and 2)
########################################################################"

#1) Pension caregiver reform
pρ = co.setup();pρ.Pmax=10000.145;pρ.add_points_exp=1.0;pρ.add_points=1.5#point_equivalent
Modρ = sol.solveEulerEquation(pρ,model='pension reform')
Sρ= sim.simNoUncer_interp(pρ,Modρ,Tstart=beg,Astart=SB['A'][beg,:],Pstart=SB['p'][beg,:],izstart=SB['iz'][beg,:])

#2) Increase in gross wages
pτ = co.setup();pτ.w[beg:end]=(1.0+increase)*p.w[beg:end]
Modτ = sol.solveEulerEquation(pτ,model='baseline')
Sτ= sim.simNoUncer_interp(pτ,Modτ,Tstart=beg,Astart=SB['A'][beg,:],Pstart=SB['p'][beg,:],izstart=SB['iz'][beg,:])


#3) Compute elasticities associated with 1) and 2)
ϵρ = (co.hours(pρ,Sρ,beg,end)/co.hours(p,SB,beg,end)-1)/increase
ϵτ = (co.hours(pτ,Sτ,beg,end)/co.hours(p,SB,beg,end)-1)/increase

print("The forward elasticity is {}, the standard elasticity is {}".format(ϵρ,ϵτ))

######################################################################"
# Below, compute marshallian, hicksian and firsh elasticities 
# 1) Marhsall
# 2) Hicks
# 3) Frisch
# 4) Display 1),2) and 3)
########################################################################

###!!! should have a loop with a change per age and then average

increase_small = increase
# 1) Marhsall: perturb the whole profile of earnings
pM = co.setup();pM.w[beg:]=(1.0+increase_small)*p.w[beg:]
ModM = sol.solveEulerEquation(pM,model='baseline')
SM= sim.simNoUncer_interp(pM,ModM,Tstart=beg,Astart=SB['A'][beg,:],Pstart=SB['p'][beg,:],izstart=SB['iz'][beg,:])

# 2) Hicks elasticity
pH = co.setup();pH.w[end]=(1.0+increase_small)*p.w[end]
ModH = sol.solveEulerEquation(pH,model='baseline')
SH= sim.simNoUncer_interp(pH,ModH,Tstart=end,Astart=SB['A'][end,:],Pstart=SB['p'][end,:],izstart=SB['iz'][end,:])

# 3) Frisch elasticity
pF = co.setup();pF.w[end]=(1.0+increase_small)*p.w[end]
ModF = sol.solveEulerEquation(pF,model='baseline')
SF= sim.simNoUncer_interp(pF,ModF,Tstart=beg,Astart=SB['A'][beg,:],Pstart=SB['p'][beg,:],izstart=SB['iz'][beg,:])


#4) Compute elasticities associated with 1),2) and 3)
ϵM = (co.hours(pM,SM,beg,end)/co.hours(p,SB,beg,end)-1)/increase_small
ϵH = (co.hours(pH,SH,end,end+1)/co.hours(p,SB,end,end+1)-1)/increase_small
ϵF = (co.hours(pF,SF,end,end+1)/co.hours(p,SB,end,end+1)-1)/increase_small


print("The Marshallian elasticity is {}, Hicks is {}, Frisch is {}".format(ϵM,ϵH,ϵF))


