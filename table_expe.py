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


#Lowe in t=3 only to mimin a 1% increase in net wages 
pτ = co.setup();pτ.τ=0.88*p.τ#0.83*p.τ
Modτ = sol.solveEulerEquation(pτ,model='baseline')




#Compute the Frisch 
#pΩ
#ModΩ
########################################
# simulate the model
########################################

#Baseline
SB= sim.simNoUncer_interp(p, ModB,Astart=p.startA,Pstart=np.zeros(p.N))
SP= sim.simNoUncer_interp(p, ModP,Astart=p.startA,Pstart=np.zeros(p.N))
Sτ= sim.simNoUncer_interp(pτ,Modτ,Astart=p.startA,Pstart=np.zeros(p.N))


# ########################################
# # Elements of the table
# ########################################


# #Marshallian elasticity: %change id(t)n h for an unexpected 1% increase in wage w from t=3
# ϵf_w=(np.mean(SWt['h'][:pWt.R,:])/np.mean(SB['h'][:pWt.R,:])-1)*100
# ϵf_τ=(np.mean(Sτt['h'][:pτt.R,:])/np.mean(SB['h'][:pτt.R,:])-1)*100

#Welfare
welf_B=np.mean(SB['v'][0,:]-1)*100
welf_P=np.mean(SP['v'][0,:]-1)*100
welf_τ=np.mean(Sτ['v'][0,:]-1)*100

#Women's labor supply
wlsB=np.mean(p.wls[SB['h'][:p.R,:]])
wlsP=np.mean(p.wls[SP['h'][:p.R,:]])
wlsτ=np.mean(p.wls[Sτ['h'][:p.R,:]])

#Govt. budget
expe_P=p.ρ*SP['p'][p.R:,:]
expe_B=p.ρ*SB['p'][p.R:,:]
expe_τ=pτ.ρ*Sτ['p'][pτ.R:,:]
tax_P=p.τ[:p.R,None]*p.wls[SP['h'][:p.R,:]]*SP['wh'][:p.R,:]
tax_B=p.τ[:p.R,None]*p.wls[SB['h'][:p.R,:]]*SB['wh'][:p.R,:]
tax_τ=pτ.τ[:pτ.R,None]*p.wls[Sτ['h'][:pτ.R,:]]*Sτ['wh'][:pτ.R,:]


#adjusted deficits
adjust=np.ones(SP['c'].shape)/((1+p.r)**(np.cumsum(np.ones(p.T))-1.0))[:,None]
deficit_B=(np.mean(expe_B*adjust[p.R:,:])-np.mean(tax_B*adjust[:p.R,:]))
deficit_P=(np.mean(expe_P*adjust[p.R:,:])-np.mean(tax_P*adjust[:p.R,:]))
deficit_τ=(np.mean(expe_τ*adjust[p.R:,:])-np.mean(tax_τ*adjust[:p.R,:]))
