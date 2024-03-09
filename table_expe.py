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

pτ = co.setup();pτ.τ[8:12]=p.τ[8:12]-0.0715
Modτ = sol.solveEulerEquation(pτ,model='baseline')

pPN = co.setup();pPN.Pmax=1000000;pPN.add_points=1.3555
ModPN = sol.solveEulerEquation(pPN,model='pension reform')

pm = co.setup();pm.wls_point=0;pm.add_points=1.19
Modm = sol.solveEulerEquation(pm,model='pension reform')

########################################
# simulate the models
########################################

SB = sim.simNoUncer_interp(p,  ModB, Astart=p.startA,Pstart=np.ones(p.N)*p.startP)
SP = sim.simNoUncer_interp(p,  ModP,Tstart=8,Astart=SB['A'][8,:],Pstart=SB['p'][8,:])
Sτ = sim.simNoUncer_interp(pτ, Modτ,Tstart=8,Astart=SB['A'][8,:],Pstart=SB['p'][8,:])
SPN= sim.simNoUncer_interp(pPN,ModPN,Tstart=8,Astart=SB['A'][8,:],Pstart=SB['p'][8,:])
Sm = sim.simNoUncer_interp(pm, Modm,Tstart=8,Astart=SB['A'][8,:],Pstart=SB['p'][8,:])


########################################
# EXPERIMENTS
########################################


adjustb=np.ones(SB['c'].shape)/((1+p.δ)**(np.cumsum(np.ones(p.T))-1.0))[:,None]
t=8
EVP    =np.mean(((np.cumsum((adjustb*SP['v'])[::-1],axis=0)[::-1])*(1+p.δ)**t)[t])
EVτ    =np.mean(((np.cumsum((adjustb*Sτ['v'])[::-1],axis=0)[::-1])*(1+p.δ)**t)[t])
EVPN   =np.mean(((np.cumsum((adjustb*SPN['v'])[::-1],axis=0)[::-1])*(1+p.δ)**t)[t])
EVm    =np.mean(((np.cumsum((adjustb*Sm['v'])[::-1],axis=0)[::-1])*(1+p.δ)**t)[t])
for i in np.linspace(1.00,1.01,100):
    
    St= sim.simNoUncer_interp(p, ModB,cadjust=i,Astart=p.startA,Pstart=np.ones(p.N)*p.startP)
    EVt = np.mean(((np.cumsum((adjustb*St['v'])[::-1],axis=0)[::-1])*(1+p.δ)**t)[t])#np.nanmean(EV_time)
    Pbetter=EVt<EVP
    τbetter=EVt<EVτ
    PNbetter=EVt<EVPN
    mbetter=EVt<EVm
    
    if Pbetter: welf_P=i-1
    if τbetter: welf_τ=i-1
    if PNbetter:welf_PN=i-1
    if mbetter: welf_m=i-1
    
    if (~Pbetter) & (~τbetter) & (~PNbetter) & (~mbetter): break




#welf_τ*np.mean(SB['c'][8:])*48    
#0.0715*np.mean(SB['wh'][8:12])*4+
    

#2) Govt. budget

#pension expenditures
expe_P=p.ρ*SP['p'][p.R:,:]
expe_B=p.ρ*SB['p'][p.R:,:]
expe_τ=pτ.ρ*Sτ['p'][pτ.R:,:]
expe_PN=pPN.ρ*SPN['p'][pPN.R:,:]
expe_m=pm.ρ*Sm['p'][pm.R:,:]

#taxes
tax_P=p.τ[8:p.R,None]*SP['wh'][8:p.R,:]
tax_B=p.τ[8:p.R,None]*SB['wh'][8:p.R,:]
tax_τ=pτ.τ[8:pτ.R,None]*Sτ['wh'][8:pτ.R,:]
tax_PN=p.τ[8:pPN.R,None]*SPN['wh'][8:pPN.R,:]
tax_m=pm.τ[8:pm.R,None]*Sm['wh'][8:pm.R,:]

#adjusted deficits
adjust=np.ones(SP['c'].shape)/((1+p.r)**(np.cumsum(np.ones(p.T))-1.0))[:,None]
deficit_B=(np.nanmean(expe_B*adjust[p.R:,:])-np.nanmean(tax_B*adjust[8:p.R,:]))
deficit_P=(np.nanmean(expe_P*adjust[p.R:,:])-np.nanmean(tax_P*adjust[8:p.R,:]))
deficit_τ=(np.nanmean(expe_τ*adjust[p.R:,:])-np.nanmean(tax_τ*adjust[8:p.R,:]))
deficit_PN=(np.nanmean(expe_PN*adjust[p.R:,:])-np.nanmean(tax_PN*adjust[8:p.R,:]))
deficit_m=(np.nanmean(expe_m*adjust[p.R:,:])-np.nanmean(tax_m*adjust[8:p.R,:]))

#3) Gender wage gaps in old age
ggap_old_B=1.0-(np.nanmean(p.ρ*SB['p'][p.R:,:]))/np.nanmean(p.y_N[p.R:,:])
ggap_old_P=1.0-(np.nanmean(p.ρ*SP['p'][p.R:,:]))/np.nanmean(p.y_N[p.R:,:])
ggap_old_τ=1.0-(np.nanmean(pτ.ρ*Sτ['p'][pτ.R:,:]))/np.nanmean(pτ.y_N[p.R:,:])
ggap_old_PN=1.0-(np.nanmean(pPN.ρ*SPN['p'][pPN.R:,:]))/np.nanmean(pPN.y_N[p.R:,:])
ggap_old_m=1.0-(np.nanmean(pm.ρ*Sm['p'][pm.R:,:]))/np.nanmean(pm.y_N[p.R:,:])

#4) WLP
WLS_B=co.hours(p,SB,8,12)#  np.nanmean(SB['h'][8:12,:]>0)
WLS_P=co.hours(p,SP,8,12)#np.nanmean(SP['h'][8:12,:]>0)
WLS_τ=co.hours(p,Sτ,8,12)#np.nanmean(Sτ['h'][8:12,:]>0)
WLS_PN=co.hours(p,SPN,8,12)#=np.nanmean(SPN['h'][8:12,:]>0)
WLS_m=co.hours(p,Sm,8,12)#np.nanmean(Sτ['h'][8:12,:]>0)

WLP_B=np.nanmean(SB['h'][8:12,:]>0)
WLP_P=np.nanmean(SP['h'][8:12,:]>0)
WLP_τ=np.nanmean(Sτ['h'][8:12,:]>0)
WLP_PN=np.nanmean(SPN['h'][8:12,:]>0)
WLP_m=np.nanmean(Sm['h'][8:12,:]>0)


#5) mini-jobs
mini_B=np.nanmean(SB['h'][8:12,:]==1)
mini_P=np.nanmean(SP['h'][8:12,:]==1)
mini_τ=np.nanmean(Sτ['h'][8:12,:]==1)
mini_PN=np.nanmean(SPN['h'][8:12,:]==1)
mini_m=np.nanmean(Sm['h'][8:12,:]==1)

# Table with experiments
def p42(x): return str('%4.2f' % x) 
def p43(x): return str('%4.2f' % x)    
def p50(x): return str('%4.2f' % x)      
table=r'\begin{table}[htbp]'+\
       r'\caption{Lifecycle model: counterfactual experiments}\label{table:experiments}'+\
       r'\centering\footnotesize'+\
       r'\begin{tabular}{lccccc}'+\
       r' \toprule '+\
       r"& Pension & Women's labor & Women's labor & Women in &  Welfare gains  \\"+\
       r"&gender gap &hours &  participation  (\%) & marginal jobs (\%)  & wrt baseline (\%)  \\"+\
       r'\midrule   '+\
       r' Baseline                                   &'+p43(ggap_old_B) +'&'+p43(WLS_B) +'&'+p43(WLP_B*100) +'&'+p43(mini_B*100) +'& 0.0\\\\'+\
       r' Caregiver credits                          &'+p43(ggap_old_P) +'&'+p43(WLS_P) +'&'+p43(WLP_P*100) +'&'+p43(mini_P*100) +'&'+p50(welf_P*100)+'\\\\'+\
       r' Lower income taxes                         &'+p43(ggap_old_τ) +'&'+p43(WLS_τ) +'&'+p43(WLP_τ*100) +'&'+p43(mini_τ*100) +'&'+p50(welf_τ*100)+'\\\\'+\
       r' Pension points + taxes for mini-jobs&'+p43(ggap_old_m) +'&'+p43(WLS_m) +'&'+p43(WLP_m*100) +'&'+p43(mini_m*100) +'&'+p50(welf_m*100)+'\\\\'+\
       r' Caregiver credits, no upper threshold      &'+p43(ggap_old_PN)+'&'+p43(WLS_PN)+'&'+p43(WLP_PN*100)+'&'+p43(mini_PN*100)+'&'+p50(welf_PN*100)+'\\\\'+\
       r' \bottomrule'+\
       r'\multicolumn{5}{l}{\textsc{Notes:} The experiments in the last three rows imply the same government deficit.}'+'\\\\'+\
       r'\multicolumn{5}{l}{Welfare gains = increase in consumption at baseline to be indifferent with the experiment under analysis. }'+\
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