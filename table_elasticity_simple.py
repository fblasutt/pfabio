# Fabio: life cycle model of consumption, savings anf FLS
#        It also features a separate budget for pension benefits,
#        following the reform of the German pension system


######################################
# Preamble
######################################


########################
#ISSUES!!!
# Discounting matters! When estimating, you get similar hrs response.
#translater into elasticity, you need to discount future income. This
#matters a lot for the results...
# Back of the envelop calculations given insanely high elasticity:
# (p.ρ*0.26*0.5*adjust[p.R:,0]).sum()*1000*(1+p.r)**8 for income increase per year
############################


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

increase=1.00

beg=8
end=12
#Baseline######################################################################

p.tax[:] = increase;ModB = sol.solveEulerEquation(p,model='baseline')
SB= sim.simNoUncer_interp(p,ModB,Astart=p.startA,Pstart=np.ones(p.N)*p.startP)
adjust=np.ones(SB['c'].shape)/((1+p.r)**(np.cumsum(np.ones(p.T))-1.0))[:,None]





#Higher pension points
pρ = co.setup();pρ.Pmax=10000.145;pρ.add_points=1.5#point_equivalent
date=beg
Modρ = sol.solveEulerEquation(pρ,model='pension reform')
Sρ= sim.simNoUncer_interp(pρ,Modρ,Tstart=date,Astart=SB['A'][date,:],Pstart=SB['p'][date,:])

##############################################"
##############################################"
##############################################"

#Elasticities computation
add_points_temp =  (1-p.τ[0])*p.ρ*(SB['wh'][beg:end ,:]*(SB['h'][beg:end ,:]>0)/p.E_bar_now)
net_income_after_ret = (add_points_temp.sum(axis=0)*adjust[p.R:,:]).sum()


#substract actual taxes to normal taxes if women earned zero to get net women's income
net_income_before_ret=np.sum((SB['wh'][beg:end ,:]*(SB['h'][beg:end ,:]>1)-(SB['taxes'][beg:end,:]-SB['taxes_mod'][beg:end,:])*(SB['h'][beg:end ,:]>1))*adjust[beg:end,:]) 
income_before=net_income_before_ret+net_income_after_ret*0 
income_after= net_income_before_ret+net_income_after_ret*0.5 
 
ϵρ = (co.hours(p,Sρ,beg,end)/co.hours(p,SB,beg,end)-1)/(income_after/income_before-1) 
ϵρ1=(np.log((co.hours(p,Sρ,beg,end)))-np.log((co.hours(p,SB,beg,end))))/np.log(income_after/income_before) 
 
