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

#Models: pension regorm (P), baseline (B), lower taxes(τ)
ModP= sol.solveEulerEquation(p,model='pension reform')
ModB = sol.solveEulerEquation(p,model='baseline')
pτ = co.setup();pτ.τ[3:11]=p.τ[3:11]-0.088
Modτ = sol.solveEulerEquation(pτ,model='baseline')

########################################
# simulate the models
########################################

SB= sim.simNoUncer_interp(p, ModB,Astart=p.startA,Pstart=np.ones(p.N)*p.startP)
SP= sim.simNoUncer_interp(p, ModP,Astart=p.startA,Pstart=np.ones(p.N)*p.startP)
Sτ= sim.simNoUncer_interp(pτ,Modτ,Astart=p.startA,Pstart=np.ones(p.N)*p.startP)


########################################
# EXPERIMENTS
########################################


#1) Welfare effects in the model, measured as wealth
for i in np.linspace(0.0,10.0,100):
    St= sim.simNoUncer_interp(p, ModB,Astart=p.startA+i,Pstart=np.ones(p.N)*p.startP)
    if np.mean(St['v'][0,:]-1)*100>np.mean(SP['v'][0,:]-1)*100:
        welf_P=i*p.scale
        break
    
for i in np.linspace(0.0,10.0,100):
    St= sim.simNoUncer_interp(p, ModB,Astart=p.startA+i,Pstart=np.ones(p.N)*p.startP)
    if np.mean(St['v'][0,:]-1)*100>np.mean(Sτ['v'][0,:]-1)*100:
        welf_τ=i*p.scale
        break
    
#2) Govt. budget

#pension expenditures
expe_P=p.ρ*SP['p'][p.R:,:]
expe_B=p.ρ*SB['p'][p.R:,:]
expe_τ=pτ.ρ*Sτ['p'][pτ.R:,:]

#taxes
tax_P=p.τ[:p.R,None]*p.wls[SP['h'][:p.R,:]]*SP['wh'][:p.R,:]
tax_B=p.τ[:p.R,None]*p.wls[SB['h'][:p.R,:]]*SB['wh'][:p.R,:]
tax_τ=pτ.τ[:pτ.R,None]*p.wls[Sτ['h'][:pτ.R,:]]*Sτ['wh'][:pτ.R,:]

#adjusted deficits
adjust=np.ones(SP['c'].shape)/((1+p.r)**(np.cumsum(np.ones(p.T))-1.0))[:,None]
deficit_B=(np.mean(expe_B*adjust[p.R:,:])-np.mean(tax_B*adjust[:p.R,:]))
deficit_P=(np.mean(expe_P*adjust[p.R:,:])-np.mean(tax_P*adjust[:p.R,:]))
deficit_τ=(np.mean(expe_τ*adjust[p.R:,:])-np.mean(tax_τ*adjust[:p.R,:]))

#3) Gender wage gaps in old age
ggap_old_B=1.0-(np.mean(p.ρ*SB['p'][p.R:,:]))/np.mean(p.y_N[p.R:,:])
ggap_old_P=1.0-(np.mean(p.ρ*SP['p'][p.R:,:]))/np.mean(p.y_N[p.R:,:])
ggap_old_τ=1.0-(np.mean(pτ.ρ*Sτ['p'][pτ.R:,:]))/np.mean(pτ.y_N[p.R:,:])

#4) WLP
WLP_B=np.mean(SB['h'][3:11,:]>0)
WLP_P=np.mean(SP['h'][3:11,:]>0)
WLP_τ=np.mean(Sτ['h'][3:11,:]>0)

# Table with experiments
def p42(x): return str('%4.2f' % x) 
def p43(x): return str('%4.3f' % x)    
def p50(x): return str('%5.0f' % x)      
table=r'\begin{table}[htbp]'+\
       r'\caption{Lifecycle model: counterfactual experiments}\label{table:experiments}'+\
       r'\centering\footnotesize'+\
       r'\begin{tabular}{lccc}'+\
       r' \toprule '+\
       r"& Pension & Women's labor & Welfare gains  \\"+\
       r"&gender gap &participation & wrt baseline (euros)  \\"+\
       r'\midrule   '+\
       r' Baseline                 &'+p43(ggap_old_B)+'&'+p43(WLP_B)+'& 0.0\\\\'+\
       r' Caregiver credits        &'+p43(ggap_old_P)+'&'+p43(WLP_P)+'&'+p50(welf_P)+'\\\\'+\
       r' Lower income taxes              &'+p43(ggap_old_τ)+'&'+p43(WLP_τ)+'&'+p50(welf_τ)+'\\\\'+\
       r'  \bottomrule'+\
       r'\multicolumn{4}{l}{\textsc{Notes:} The experiments in the last two rows imply the same government deficit}'+'\\\\'+\
       r'\multicolumn{4}{l}{of '+p50((deficit_P-deficit_B)*p.scale)+' euros. Welfare gains = equivalent transfer in baseline model at age 30. }'+\
      """\end{tabular}
      """+\
      r'\end{table}'
      
#Write table to tex file
with open('C:/Users/Fabio/Dropbox/occupation/model/pfabio/output/table_expe.tex', 'w') as f:
    f.write(table)
    f.close()


###################################
# TARGETED MOMENTS AND PRAMETERS
####################################

#1) effect of the reform on employment
eff_empl=np.mean(p.wls[SP['h'][3:11,:]]>0)-np.mean(p.wls[SB['h'][3:11,:]]>0)

#2) employment at baseline
employment_rate=np.mean(SB['h'][3:11,:]>0)

#3) fulltime employment
work_fulltime=np.mean(SB['h'][3:11,:]==1)

#Table with parameters    
table=r'\begin{table}[htbp]'+\
       r'\caption{Model parameters and fit}\label{table:model_param}'+\
       r'\centering\footnotesize'+\
       r'\begin{tabular}{lcccc}'+\
       r' \toprule '+\
       r" Parameter & Value & \multicolumn{3}{c}{Target statistics}  \\\cline{3-5} "+\
       r" &  &  Name & Data & Model  \\"+\
       r'\midrule   '+\
       r' Discount rate ($\delta$)      &'+p42(p.δ*100)+'\%& Effect of reform on employment   & 0.07 &'+p42(eff_empl)+'\\\\'+\
       r' Fixed cost of working ($q$)   &'+p42(p.q)+'& Employment rate                  & 0.65 &'+p42(employment_rate)+'\\\\'+\
       r' Weight on leisure ($\beta$)   &'+p42(p.β)+'& Work full time             & 0.31 &'+p42(work_fulltime)+'\\\\'+\
       r'  \bottomrule'+\
      """\end{tabular}"""+\
      r'\end{table}'
      
#Write table to tex file
with open('C:/Users/Fabio/Dropbox/occupation/model/pfabio/output/table_params.tex', 'w') as f:
    f.write(table)
    f.close()
   
###############################
#Compute nontargeted moments
##############################

#MPE out of pension wealth
Sw= sim.simNoUncer_interp(p,ModB,Astart=p.startA+100/p.scale,Pstart=np.ones(p.N)*p.startP)
MPE=((np.mean(p.wls[Sw['h'][1:3,:]]*Sw['wh'][1:3,:])-np.mean(p.wls[SB['h'][1:3,:]]*SB['wh'][1:3,:]))*p.scale)

#Effect of pension reform on:
    
#1) pension points
eff_points=np.mean(np.diff(SP['p'][3:11,:],axis=0))-np.mean(np.diff(SB['p'][3:11,:],axis=0))

#2) full employment
eff_earn=(np.mean(p.wls[SP['h'][3:11,:]]*SP['wh'][3:11,:])-np.mean(p.wls[SB['h'][3:11,:]]*SB['wh'][3:11,:]))*p.scale

#3) earnings
eff_emp=np.mean(SP['h'][3:11,:][SP['h'][3:11,:]>0]==2)-np.mean(SB['h'][3:11,:][SB['h'][3:11,:]>0]==2)

############################################
#Table with parameters
###########################################     
table=r'\begin{table}[htbp]'+\
       r'\caption{Non-targeted moments}\label{table:nontargeted_moments}'+\
       r'\centering\footnotesize'+\
       r'\begin{tabular}{lccc}'+\
       r' \toprule '+\
       r" &  Source& Data & Model  \\"+\
       r'\midrule   '+\
       r' Wealth effect on earnings      & Artmann et al. (2023)          & -5.1 &'+p42(MPE)+'\\\\'+\
       r' Pension points                & Data                           & 0.15 &'+p42(eff_points)+'\\\\'+\
       r' Work full time              & Data                             & 0.045 &'+p42(eff_emp)+'\\\\'+\
       r'  \bottomrule'+\
      """\end{tabular}"""+\
      r'\end{table}'
      
#Write table to tex file
with open('C:/Users/Fabio/Dropbox/occupation/model/pfabio/output/table_nontargetd.tex', 'w') as f:
    f.write(table)
    f.close()