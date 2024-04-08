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


#Baseline
ModB = sol.solveEulerEquation(p,model='baseline')
SB= sim.simNoUncer_interp(p,ModB,Astart=p.startA,Pstart=np.ones(p.N)*p.startP)

#Taxes
increase=1.01
date=0
#Wages 1% hihgher than baseline in t=3 only + baseline
#pτ = co.setup();pτ.w=pτ.w*increase
pτ = co.setup();pτ.τ=p.τ-0.01#1-(1-p.τ)*increase
Modτ = sol.solveEulerEquation(pτ,model='baseline')
Sτ= sim.simNoUncer_interp(pτ,Modτ,Tstart=date,Astart=SB['A'][date,:],Pstart=SB['p'][date,:])

#Pension
pρ = co.setup();pρ.points_base=1.145
#pρ = co.setup();pρ.w=pρ.w*increase
Modρ = sol.solveEulerEquation(pρ,model='baseline')
Sρ= sim.simNoUncer_interp(pρ,Modρ,Tstart=date,Astart=SB['A'][date,:],Pstart=SB['p'][date,:])

##############################################"
##############################################"
##############################################"



expe_τ=pτ.ρ*Sτ['p'][pτ.R:,:]
expe_ρ=pρ.ρ*Sρ['p'][pρ.R:,:]

#taxes

#aaa1=(0.01*pτ.τ[:p.R,None]*SB['wh'][:p.R,:]*adjust[:p.R,:]).sum()
#aaa2=np.nansum(0.0108*p.ρ*SB['p'][p.R:,:]*adjust[p.R:,:])

tax_τ=pτ.τ[:pτ.R,None]*Sτ['wh'][:pτ.R,:]
tax_ρ=pρ.τ[:pρ.R,None]*Sρ['wh'][:pρ.R,:]

#adjusted deficits
adjust=np.ones(SB['c'].shape)/((1+p.r)**(np.cumsum(np.ones(p.T))-1.0))[:,None]

deficit_τ=(np.nansum(expe_τ*adjust[p.R:,:])  -np.nansum(tax_τ*adjust[:p.R,:]))
deficit_ρ=(np.nansum(expe_ρ*adjust[p.R:,:])  -np.nansum(tax_ρ*adjust[:p.R,:]))

##############################################"
##############################################"
##############################################"

#Elasticities
ϵτ=(co.hours(p,Sτ,date,p.R)/co.hours(p,SB,date,p.R)-1)*100
ϵρ=(co.hours(p,Sρ,date,p.R)/co.hours(p,SB,date,p.R)-1)*100

#(np.log((co.hours(p,Sτ,date,p.R)))-np.log((co.hours(p,SB,date,p.R))))/np.log((1-p.τ[0]+.01)/(1-p.τ[0]))
#(np.log((Sτ['h'][:p.R]>0).mean())-np.log((SB['h'][:p.R]>0).mean()))/np.log((1-p.τ[0]+.01)/(1-p.τ[0]))

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
