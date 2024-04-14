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
#too many women reduce WLS in actual experiment: when you take out threshold you
#get insane response. Earnings should be in line with data!
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

increase=0.01

beg=8
end=p.R
#Baseline######################################################################

p.tax[:] = -increase;ModB = sol.solveEulerEquation(p,model='baseline')
SB= sim.simNoUncer_interp(p,ModB,Astart=p.startA,Pstart=np.ones(p.N)*p.startP)
adjust=np.ones(SB['c'].shape)/((1+p.r)**(np.cumsum(np.ones(p.T))-1.0))[:,None]

#compute increase in net income if all wages go up by 1%, keeping behavior constant
net_income_increase_before_ret = np.sum((SB['wh'][beg:end ,:]*(1.0+increase)-SB['taxes_mod'][beg:end ,:])*adjust[beg:end,:]) 
net_income_before_ret =          np.sum((SB['wh'][beg:end ,:]               -SB['taxes'][beg:end ,:])*adjust[beg:end,:]) 


add_points_temp =  (1-p.τ[0])*p.ρ*(SB['wh'][beg:end ,:]*(SB['h'][beg:end ,:]>0)/p.E_bar_now)

net_income_after_ret = (add_points_temp.sum(axis=0)*adjust[p.R:,:]).sum()


#net_income_after_ret = (1-p.τ[0])*np.sum(p.ρ*(SB['p'][p.R: ,:]-p.startP)*adjust[p.R:,:])

#compute the increase in points necessary to have the same increase in net income
#note that we account for the increase in points due to higher wages
share=(end-beg)/p.R
point_equivalent = (net_income_increase_before_ret+(1.0+increase)*net_income_after_ret-net_income_before_ret)/(net_income_after_ret)


#Higher gross wages
date=beg
pτ = co.setup();pτ.w[beg:end,:,:]=p.w[beg:end,:,:]*(1.0+increase)
Modτ = sol.solveEulerEquation(pτ,model='baseline')
Sτ= sim.simNoUncer_interp(pτ,Modτ,Tstart=date,Astart=SB['A'][date,:],Pstart=SB['p'][date,:])

#Higher pension points
pρ = co.setup();pρ.Pmax=10000.145;pρ.points_base=point_equivalent

Modρ = sol.solveEulerEquation(pρ,model='baseline')
Sρ= sim.simNoUncer_interp(pρ,Modρ,Tstart=date,Astart=SB['A'][date,:],Pstart=SB['p'][date,:])

##############################################"
##############################################"
##############################################"

#Elasticities

ϵτ=(co.hours(p,Sτ,beg,end)/co.hours(p,SB,beg,end)-1)*100
ϵρ=(co.hours(p,Sρ,beg,end)/co.hours(p,SB,beg,end)-1)*100

ϵτ1=(np.log((co.hours(p,Sτ,date,p.R)))-np.log((co.hours(p,SB,date,p.R))))/np.log((1-pτ.tax)/(1-p.tax))
ϵρ1=(np.log((co.hours(p,Sρ,date,p.R)))-np.log((co.hours(p,SB,date,p.R))))/np.log((1-pτ.tax)/(1-p.tax))

#(np.log((co.hours(p,Sτ,date,p.R)))-np.log((co.hours(p,SB,date,p.R))))/np.log((1-pτ.τ[0])/(1-p.τ[0]))
#(np.log((Sτ['h'][:p.R]>0).mean())-np.log((SB['h'][:p.R]>0).mean()))/np.log((1-pτ.τ[0])/(1-p.τ[0]))

# from sklearn.linear_model import LinearRegression
# y=np.log(co.hours_pr(p,SB,date,p.R).flatten())
# x=(np.log(SB['w'][:p.R])).flatten().reshape((-1, 1))


# #y=np.diff(np.log(co.hours_pr(p,SB,date,p.R)),axis=0).flatten()
# #x=np.diff((np.log(SB['w'][:p.R])),axis=0).flatten().reshape((-1, 1))

# #y=np.log(co.hours_pr(p,SB,date,p.R).flatten())
# #x=((np.log(SB['w'][:p.R])).flatten()-np.log(SB['c'][:p.R]).flatten()).reshape((-1, 1))

# #y=np.log(co.hours_pr(p,Sτ,date,p.R).flatten())
# #x=(np.log(Sτ['w'][:p.R])).flatten().reshape((-1, 1))

# model = LinearRegression().fit(x, y)
# aaa=model.coef_

# def p31(x): return str('%4.3f' % x)    
# table=r'\begin{table}[htbp]'+\
#         r'\caption{Elasticities of labor supply using different experiments}\label{table:elasticities}'+\
#         r'\centering\footnotesize'+\
#         r'\begin{tabular}{lcc}'+\
#         r' \toprule '+\
#         r"& Change in net wages & Change in pension points   \\"+\
#         r'\midrule   '+\
#         r' Elasticity of labor supply &'+p31(ϵτ)+'&'+p31(ϵρ)+'\\\\'+\
#         r'  \bottomrule'+\
#       """\end{tabular}
#       """+\
#       r'\end{table}'
 
# # r'\multicolumn{3}{l}{\textsc{Notes:} The change in pension points make the two reforms revenue neutral.}'+\
# #Write table to tex file
# with open('C:/Users/32489/Dropbox/occupation/model/pfabio/output/table_elasticities.tex', 'w') as f:
#     f.write(table)
#     f.close()
