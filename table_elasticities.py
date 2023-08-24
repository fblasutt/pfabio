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

#Wages 1% hihgher than baseline in t=3 only + baseline
pWtB = co.setup();pWtB.w=1.01*pWtB.w
ModWtB = sol.solveEulerEquation(pWtB,model='baseline')

#Lowe in t=3 only to mimin a 1% increase in net wages + baseline
pτtB = co.setup();pτtB.τ=1.01*p.τ-0.01#1-(1-p.τ[3])*1.01
ModτtB = sol.solveEulerEquation(pτtB,model='baseline')


#Pension reform
ModP = sol.solveEulerEquation(p,model='pension reform')

#Wages 1% hihgher than baseline in t=3 only + baseline
pWtP = co.setup();pWtP.w=1.01*pWtP.w
ModWtP = sol.solveEulerEquation(pWtP,model='pension reform')

#Lowe in t=3 only to mimin a 1% increase in net wages + baseline
pτtP = co.setup();pτtP.τ=1.01*p.τ-0.01#1-(1-p.τ[3])*1.01
ModτtP = sol.solveEulerEquation(pτtP,model='pension reform')


#Compute the Frisch 
#pΩ
#ModΩ
########################################
# simulate the model
########################################

#Baseline
SB= sim.simNoUncer_interp(p,ModB,Astart=p.startA,Pstart=np.zeros(p.N))

#Wages 1% higher than baseline in t=3 only + baseline
SWtB= sim.simNoUncer_interp(pWtB,ModWtB,Astart=p.startA,Pstart=np.zeros(p.N))

# Lower taxes in t=3 only  + baseline
SτtB= sim.simNoUncer_interp(pτtB,ModτtB,Astart=p.startA,Pstart=np.zeros(p.N))

#Baseline
SP= sim.simNoUncer_interp(p,ModP,Astart=p.startA,Pstart=np.zeros(p.N))

#Wages 1% higher than baseline in t=3 only + pension reform
SWtP= sim.simNoUncer_interp(pWtP,ModWtP,Astart=p.startA,Pstart=np.zeros(p.N))

# Lower taxes in t=3 only  + pension reform
SτtP= sim.simNoUncer_interp(pτtP,ModτtP,Astart=p.startA,Pstart=np.zeros(p.N))


# ########################################
# # Elements of the table
# ########################################

#Marshallian elasticity: %change id(t)n h for an unexpected 1% increase in wage w from t=3
ϵf_wB=(np.mean(SWtB['h'][:pWtB.R,:])/np.mean(SB['h'][:pWtB.R,:])-1)*100
ϵf_τB=(np.mean(SτtB['h'][:pτtB.R,:])/np.mean(SB['h'][:pτtB.R,:])-1)*100

ϵf_wP=(np.mean(SWtP['h'][:pWtP.R,:])/np.mean(SP['h'][:pWtP.R,:])-1)*100
ϵf_τP=(np.mean(SτtP['h'][:pτtP.R,:])/np.mean(SP['h'][:pτtP.R,:])-1)*100




#construct the table

      
def p31(x): return str('%4.3f' % x)    
table=r'\begin{table}[htbp]'+\
       r'\caption{Elasticities of labor supply using different experiments}\label{table:elasticities}'+\
       r'\centering\footnotesize'+\
       r'\begin{tabular}{lcc}'+\
       r' \toprule '+\
       r"& Change in gross wages & Change in tax rate   \\"+\
       r'\midrule   '+\
       r' Marshallian elasticity of labor supply &'+p31(ϵf_wB)+'&'+p31(ϵf_τB)+'\\\\'+\
       r'  \bottomrule'+\
       r'\multicolumn{3}{l}{\textsc{Notes:} All the elsticities are computed for the same change in net wages.}'+\
      """\end{tabular}
      """+\
      r'{\raggedright\footnotesize \textsc{Notes:} All the elsticities are computed for the same change in net wages.\par}'+\
      r'\end{table}'
      
#Write table to tex file
with open('C:/Users/Fabio/Dropbox/occupation/model/pfabio/output/table_expe.tex', 'w') as f:
    f.write(table)
    f.close()
