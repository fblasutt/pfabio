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

#Models: pension reform (P), baseline (B), lower taxes(τ), pension reform without limit (PN)
ModP= sol.solveEulerEquation(p,model='pension reform')
ModB = sol.solveEulerEquation(p,model='baseline')

pτ = co.setup();pτ.τ[8:12]=p.τ[8:12]-0.086
Modτ = sol.solveEulerEquation(pτ,model='baseline')

pPN = co.setup();pPN.Pmax=1000000;pPN.add_points=1.43
ModPN = sol.solveEulerEquation(pPN,model='pension reform')

########################################
# simulate the models
########################################

SB = sim.simNoUncer_interp(p,  ModB, Astart=p.startA,Pstart=np.ones(p.N)*p.startP)
SP = sim.simNoUncer_interp(p,  ModP,Tstart=8,Astart=SB['A'][8,:],Pstart=SB['p'][8,:])
Sτ = sim.simNoUncer_interp(pτ, Modτ,Tstart=8,Astart=SB['A'][8,:],Pstart=SB['p'][8,:])
SPN= sim.simNoUncer_interp(pPN,ModPN,Tstart=8,Astart=SB['A'][8,:],Pstart=SB['p'][8,:])


########################################
# EXPERIMENTS
########################################


#1) Welfare effects in the model, measured as wealth
# for i in np.linspace(2.0,4.0,100):
#     St= sim.simNoUncer_interp(p, ModB,Tstart=8,Astart=SB['A'][8,:]+i,Pstart=SB['p'][8,:])
    
#     Pbetter=np.nanmean(St['ev'][8,:]-1)*100<np.nanmean(SP['ev'][8,:]-1)*100
#     τbetter=np.nanmean(St['ev'][8,:]-1)*100<np.nanmean(Sτ['ev'][8,:]-1)*100
#     PNbetter=np.nanmean(St['ev'][8,:]-1)*100<np.nanmean(SPN['ev'][8,:]-1)*100
    
#     if Pbetter: welf_P=i*p.scale   
#     if τbetter: welf_τ=i*p.scale
#     if PNbetter:welf_PN=i*p.scale
    
#     if (~Pbetter) & (~τbetter) & (~PNbetter): break


adjust=np.ones(SB['c'].shape)/((1+p.δ)**(np.cumsum(np.ones(p.T))-1.0))[:,None]
for i in np.linspace(1.01,1.03,100):
    St= sim.simNoUncer_interp(p, ModB,cadjust=i)
    EV=(np.cumsum((adjust*St['v'])[::-1],axis=0)[::-1])
    EV_time=EV[8]*(1+p.δ)**8
    
    Pbetter=np.nanmean(EV_time)<np.nanmean(SP['ev'][8,:])
    τbetter=np.nanmean(EV_time)<np.nanmean(Sτ['ev'][8,:])
    PNbetter=np.nanmean(EV_time)<np.nanmean(SPN['ev'][8,:])
    
    if Pbetter: welf_P=i-1
    if τbetter: welf_τ=i-1
    if PNbetter:welf_PN=i-1
    
    if (~Pbetter) & (~τbetter) & (~PNbetter): break

    
    

#2) Govt. budget

#pension expenditures
expe_P=p.ρ*SP['p'][p.R:,:]
expe_B=p.ρ*SB['p'][p.R:,:]
expe_τ=pτ.ρ*Sτ['p'][pτ.R:,:]
expe_PN=pPN.ρ*SPN['p'][pPN.R:,:]

#taxes
tax_P=p.τ[8:p.R,None]*SP['wh'][8:p.R,:]
tax_B=p.τ[8:p.R,None]*SB['wh'][8:p.R,:]
tax_τ=pτ.τ[8:pτ.R,None]*Sτ['wh'][8:pτ.R,:]
tax_PN=p.τ[8:pPN.R,None]*SPN['wh'][8:pPN.R,:]

#adjusted deficits
adjust=np.ones(SP['c'].shape)/((1+p.r)**(np.cumsum(np.ones(p.T))-1.0))[:,None]
deficit_B=(np.nanmean(expe_B*adjust[p.R:,:])-np.nanmean(tax_B*adjust[8:p.R,:]))
deficit_P=(np.nanmean(expe_P*adjust[p.R:,:])-np.nanmean(tax_P*adjust[8:p.R,:]))
deficit_τ=(np.nanmean(expe_τ*adjust[p.R:,:])-np.nanmean(tax_τ*adjust[8:p.R,:]))
deficit_PN=(np.nanmean(expe_PN*adjust[p.R:,:])-np.nanmean(tax_PN*adjust[8:p.R,:]))

#3) Gender wage gaps in old age
ggap_old_B=1.0-(np.nanmean(p.ρ*SB['p'][p.R:,:]))/np.nanmean(p.y_N[p.R:,:])
ggap_old_P=1.0-(np.nanmean(p.ρ*SP['p'][p.R:,:]))/np.nanmean(p.y_N[p.R:,:])
ggap_old_τ=1.0-(np.nanmean(pτ.ρ*Sτ['p'][pτ.R:,:]))/np.nanmean(pτ.y_N[p.R:,:])
ggap_old_PN=1.0-(np.nanmean(pPN.ρ*SPN['p'][pPN.R:,:]))/np.nanmean(pPN.y_N[p.R:,:])

#4) WLP
WLP_B=co.hours(p,SB,8,12)#  np.nanmean(SB['h'][8:12,:]>0)
WLP_P=co.hours(p,SP,8,12)#np.nanmean(SP['h'][8:12,:]>0)
WLP_τ=co.hours(p,Sτ,8,12)#np.nanmean(Sτ['h'][8:12,:]>0)
WLP_PN=co.hours(p,SPN,8,12)#=np.nanmean(SPN['h'][8:12,:]>0)

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
       r' Baseline                                &'+p43(ggap_old_B) +'&'+p43(WLP_B) +'& 0.0\\\\'+\
       r' Caregiver credits                       &'+p43(ggap_old_P) +'&'+p43(WLP_P) +'&'+p50(welf_P)+'\\\\'+\
       r' Lower income taxes                      &'+p43(ggap_old_τ) +'&'+p43(WLP_τ) +'&'+p50(welf_τ)+'\\\\'+\
       r' Caregiver credits, no upper threshold   &'+p43(ggap_old_PN)+'&'+p43(WLP_PN)+'&'+p50(welf_PN)+'\\\\'+\
       r'  \bottomrule'+\
       r'\multicolumn{4}{l}{\textsc{Notes:} The experiments in the last three rows imply the same government deficit}'+'\\\\'+\
       r'\multicolumn{4}{l}{of '+p50((deficit_P-deficit_B)*p.scale)+' euros. Welfare gains = equivalent transfer in baseline model at age 30. }'+\
      """\end{tabular}
      """+\
      r'\end{table}'
      
#Write table to tex file
with open('C:/Users/32489/Dropbox/occupation/model/pfabio/output/table_expe.tex', 'w') as f:
    f.write(table)
    f.close()


###################################
# TARGETED MOMENTS AND PRAMETERS
####################################

#1) effect of the reform on employment
eff_empl=np.nanmean(p.wls[SP['h'][8:12,:]]>0)-np.nanmean(p.wls[SB['h'][8:12,:]]>0)

#2) employment at baseline
employment_rate=np.nanmean(SB['h'][8:12,:]>0)

#3) fulltime employment
work_fulltime=np.nanmean(SB['h'][8:12,:]==1)

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
with open('C:/Users/32489/Dropbox/occupation/model/pfabio/output/table_params.tex', 'w') as f:
    f.write(table)
    f.close()
   
###############################
#Compute nontargeted moments
##############################

#MPE out of pension wealth
Sw= sim.simNoUncer_interp(p,ModB,Astart=p.startA+100/p.scale,Pstart=np.ones(p.N)*p.startP)
MPE=((np.nanmean(p.wls[Sw['h'][1:3,:]]*Sw['wh'][1:3,:])-np.nanmean(p.wls[SB['h'][1:3,:]]*SB['wh'][1:3,:]))*p.scale)

#Effect of pension reform on:
    
#1) pension points
eff_points=np.nanmean(np.diff(SP['p'][8:12,:],axis=0))-np.nanmean(np.diff(SB['p'][8:12,:],axis=0))

#2) full employment
eff_earn=(np.nanmean(p.wls[SP['h'][8:12,:]]*SP['wh'][8:12,:])-np.nanmean(p.wls[SB['h'][8:12,:]]*SB['wh'][8:12,:]))*p.scale

#3) earnings
eff_emp=np.nanmean(SP['h'][8:12,:][SP['h'][8:12,:]>0]==2)-np.nanmean(SB['h'][8:12,:][SB['h'][8:12,:]>0]==2)

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
with open('C:/Users/32489/Dropbox/occupation/model/pfabio/output/table_nontargetd.tex', 'w') as f:
    f.write(table)
    f.close()