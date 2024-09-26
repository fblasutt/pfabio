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
 
pτ = co.setup();pτ.tbase[3:11]=p.tbase[3:11]-0.25
Modτ = sol.solveEulerEquation(pτ,model='baseline') 
 
pPN = co.setup();pPN.Pmax=1000000;pPN.add_points=1.4
ModPN = sol.solveEulerEquation(pPN,model='pension reform') 
 
#pm = co.setup();pm.wls_point=0;pm.add_points=1.543 
#Modm = sol.solveEulerEquation(pm,model='pension reform') 
 
######################################## 
# simulate the models 
######################################## 
 
SB = sim.simNoUncer_interp(p,  ModB, Tstart=np.zeros(p.N,dtype=np.int16),Astart=p.startA,Pstart=np.ones((p.T,p.N))*p.startP,izstart=p.tw) 
#SB = sim.simNoUncer_interp(p,  ModB,Tstart=np.zeros(p.N,dtype=np.int16)+3,Astart=SB['A'],Pstart=SB['p'],izstart=SB['iz']) 
SP = sim.simNoUncer_interp(p,  ModP,Tstart=np.zeros(p.N,dtype=np.int16)+3,Astart=SB['A'],Pstart=SB['pb3'],izstart=SB['iz']) 
Sτ = sim.simNoUncer_interp(pτ, Modτ,Tstart=np.zeros(p.N,dtype=np.int16)+3,Astart=SB['A'],Pstart=SB['p'],izstart=SB['iz']) 
SPN= sim.simNoUncer_interp(pPN,ModPN,Tstart=np.zeros(p.N,dtype=np.int16)+3,Astart=SB['A'],Pstart=SB['p'],izstart=SB['iz']) 
#Sm = sim.simNoUncer_interp(pm, Modm,Tstart=np.zeros(p.N,dtype=np.int16)+3,Astart=SB['A'],Pstart=SB['p']) 
 
 
######################################## 
# EXPERIMENTS 
######################################## 
 
 
adjustb=np.ones(SB['c'].shape)/((1+p.δ)**(np.cumsum(np.ones(p.T))-1.0))[:,None] 
t=3
EVP    =np.mean(((np.cumsum((adjustb*SP['v'])[::-1],axis=0)[::-1])*(1+p.δ)**t)[t]) 
EVτ    =np.mean(((np.cumsum((adjustb*Sτ['v'])[::-1],axis=0)[::-1])*(1+p.δ)**t)[t]) 
EVPN   =np.mean(((np.cumsum((adjustb*SPN['v'])[::-1],axis=0)[::-1])*(1+p.δ)**t)[t]) 
#EVm    =np.mean(((np.cumsum((adjustb*Sm['v'])[::-1],axis=0)[::-1])*(1+p.δ)**t)[t]) 
for i in np.linspace(1.00005,1.0199,100): 
     
    St= sim.simNoUncer_interp(p, ModB,cadjust=i,Tstart=np.zeros(p.N,dtype=np.int16)+3,Astart=SB['A'],Pstart=SB['p'],izstart=SB['iz']) 
    EVt = np.mean(((np.cumsum((adjustb*St['v'])[::-1],axis=0)[::-1])*(1+p.δ)**t)[t])#np.nanmean(EV_time) 
    Pbetter=EVt<EVP 
    τbetter=EVt<EVτ 
    PNbetter=EVt<EVPN 
    
     
    if Pbetter: welf_P=i-1 
    if τbetter: welf_τ=i-1 
    if PNbetter:welf_PN=i-1 
     
     
    if (~Pbetter) & (~τbetter) & (~PNbetter) : break 
 
 
 
 
#welf_τ*np.mean(SB['c'][3:])*48     
#0.0715*np.mean(SB['wh'][3:11])*4+ 
     
 
#2) Govt. budget 

 
#adjusted deficits 
adjust=np.ones(SP['c'].shape)/((1+p.r)**(np.cumsum(np.ones(p.T))-1.0))[:,None] 
deficit_B=np.nansum(adjust*SB['taxes'])#(np.nansum(expe_B*adjust[p.R:,:])  -np.nansum(tax_B*adjust[3:p.R,:])) 
deficit_P=np.nansum(adjust*SP['taxes'])#(np.nansum(expe_P*adjust[p.R:,:])  -np.nansum(tax_P*adjust[3:p.R,:])) 
deficit_τ=np.nansum(adjust*Sτ['taxes'])#(np.nansum(expe_τ*adjust[p.R:,:])  -np.nansum(tax_τ*adjust[3:p.R,:])) 
deficit_PN=np.nansum(adjust*SPN['taxes'])#(np.nansum(expe_PN*adjust[p.R:,:])-np.nansum(tax_PN*adjust[3:p.R,:])) 

#3) Gender wage gaps in old age 
ggap_old_B=1.0-(np.nanmean(p.ρ*SB['p'][SB['ir']==1]))/np.nanmean(p.y_N[35,SB['iz']])
ggap_old_P=1.0-(np.nanmean(p.ρ*SP['p'][SP['ir']==1]))/np.nanmean(p.y_N[35,SB['iz']])
ggap_old_τ=1.0-(np.nanmean(pτ.ρ*Sτ['p'][Sτ['ir']==1]))/np.nanmean(pτ.y_N[35,SB['iz']])
ggap_old_PN=1.0-(np.nanmean(pPN.ρ*SPN['p'][SPN['ir']==1]))/np.nanmean(pPN.y_N[35,SB['iz']])
#ggap_old_m=1.0-(np.nanmean(pm.ρ*Sm['p'][pm.R:,:]))/np.nanmean(pm.y_N[p.R:,:]) 
 
#4) WLP 
WLS_B=co.hours(p,SB,3,32)#  np.nanmean(SB['h'][3:12,:]>0) 
WLS_P=co.hours(p,SP,3,32)#np.nanmean(SP['h'][3:12,:]>0) 
WLS_τ=co.hours(p,Sτ,3,32)#np.nanmean(Sτ['h'][3:12,:]>0) 
WLS_PN=co.hours(p,SPN,3,32)#=np.nanmean(SPN['h'][3:12,:]>0) 
#WLS_m=co.hours(p,Sm,3,12)#np.nanmean(Sτ['h'][3:12,:]>0) 
 
WLP_B=np.nanmean(SB['h'][3:32,:]>0) 
WLP_P=np.nanmean(SP['h'][3:32,:]>0) 
WLP_τ=np.nanmean(Sτ['h'][3:32,:]>0) 
WLP_PN=np.nanmean(SPN['h'][3:32,:]>0) 
#WLP_m=np.nanmean(Sm['h'][3:32,:]>0) 
 
 
#5) mini-jobs 
ret_B=np.mean(p.T-np.cumsum(SB['ir'],axis=0)[-1])
ret_P=np.mean(p.T-np.cumsum(SP['ir'],axis=0)[-1])
ret_τ=np.mean(pτ.T-np.cumsum(Sτ['ir'],axis=0)[-1])
ret_PN=np.mean(pPN.T-np.cumsum(SPN['ir'],axis=0)[-1]) 
#mini_m=np.nanmean(Sm['h'][3:32,:]==1) 
 
# Table with experiments 
def p42(x): return str('%4.2f' % x)  
def p43(x): return str('%4.3f' % x)     
def p40(x): return str('%4.0f' % x)       
table=r'\begin{table}[htbp]'+\
      r'\begin{threeparttable}'+\
       r'\caption{Lifecycle model: counterfactual experiments}\label{table:experiments}'+\
       r'\centering\footnotesize'+\
       r'\begin{tabular}{lccccc}'+\
       r' \toprule '+\
       r"& Pension & Women's labor & Women's labor & Average age &  Welfare gains  \\"+\
       r"&gender gap &hours &  participation  (\%) & at retirement  & wrt baseline (\%)  \\"+\
       r'\midrule   '+\
       r' Baseline                                   &'+p43(ggap_old_B)  +'&'+p42(WLS_B)  +'&'+p42(WLP_B*100)  +'&'+p42(ret_B) +'& 0.0\\\\'+\
       r' Caregiver credits                          &'+p43(ggap_old_P)  +'&'+p42(WLS_P)  +'&'+p42(WLP_P*100)  +'&'+p42(ret_P) +'&'+p43(welf_P*100)+'\\\\'+\
       r' Caregiver credits, no threshold            &'+p43(ggap_old_PN) +'&'+p42(WLS_PN) +'&'+p42(WLP_PN*100) +'&'+p42(ret_PN) +'&'+p43(welf_P*100)+'\\\\'+\
       r' Lower income taxes                         &'+p43(ggap_old_τ)  +'&'+p42(WLS_τ)  +'&'+p42(WLP_τ*100)  +'&'+p42(ret_τ) +'&'+p43(welf_τ*100)+'\\\\'+\
       r' \bottomrule'+\
       r'\end{tabular}'+\
       r'\begin{tablenotes}[flushleft]\small\item \textsc{Notes:} The experiments in the last three rows imply the same government deficit.'+\
       r' Welfare gains = increase in consumption at baseline to be indifferent with the experiment under analysis.'+\
       r' Reforms are in place while the child is 10 y.o. or younger''\\\\'+\
       r'\end{tablenotes}'+\
      r'\end{threeparttable}'+\
      r'\end{table}' 
       
#Write table to tex file 
with open('C:/Users/32489/Dropbox/occupation/model/pfabio/output/table_expe.tex', 'w') as f: 
    f.write(table) 
    f.close() 


################################### 
# TARGETED MOMENTS AND PRAMETERS 
#################################### 
 
#1) effect of the reform on employment 
eff_h=co.hours(p,SP,8,12)-co.hours(p,SB,8,12) 
 
#2) share into mini-jobs 
sh_mini = np.nanmean(SB['h'][7,:]==1) 
 
#3) share part-tme 
sh_part = np.nanmean(SB['h'][7,:]==2) 
 
#4) share full 
sh_full = np.nanmean(SB['h'][7,:]==3) 
 
#5) effect on employment 
eff_empl=np.nanmean(p.wls[SP['h'][8:12,:]]>0)-np.nanmean(p.wls[SB['h'][8:12,:]]>0) 
 
 
#Table with parameters     
table=r'\begin{table}[htbp]'+\
        r'\caption{Model parameters and fit}\label{table:model_param}'+\
        r'\centering\footnotesize'+\
        r'\begin{tabular}{lcccc}'+\
        r' \toprule '+\
        r" Parameter & Value & \multicolumn{3}{c}{Target statistics}  \\\cline{3-5} "+\
        r" &  &  Name & Data & Model  \\"+\
        r'\midrule   '+\
        r' Cost of working - mini ($q_{10}$)   &'+p43(p.q[1]*p.qmean)+'& Share mini-jobs           & 0.26 &'+p42(sh_mini)+'\\\\'+\
        r' Cost of working - part ($q_{20}$)   &'+p43(p.q[2]*p.qmean)+'& Share part-time           & 0.20 &'+p42(sh_part)+'\\\\'+\
        r' Cost of working - full ($q_{38.5}$)      &'+p43(p.qmean)+'& Share full time      & 0.20 &'+p42(sh_full)+'\\\\'+\
        r' Fixed effects dispersion ($\sigma_q$)   &'+p43(p.qvar)+'& \\begin{tabular}{@{}c@{}}Effect of the reform on employment \\\\ Effect of the reform on hours\\end{tabular}   & \\begin{tabular}{@{}c@{}}0.06 \\\\ 2.31\\end{tabular}& \\begin{tabular}{@{}c@{}}'+p42(eff_empl)+' \\\\'+p42(eff_h)+'\\end{tabular}\\\\'+\
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
 
def share(r,δ,T,k,per): 
    #Model-based annuitization to compute MPE: https://michael-graber.github.io/pdf/Golosov-Graber-Mogstad-Novgorodsky-2023.pdf pg 38 
    share = np.array([ ((1+r)/(1+δ))**(t+1)*(δ/(1+δ))*(1-(1/(1+δ))**(T-k))**-1  for t in range(per)]) 
     
    return share 
 
adjustr = adjust*(1+p.r)**8 
 
#MPE out of pension wealth, using tretroactive credits 
SB_retro= sim.simNoUncer_interp(p,ModB,Tstart=8,Astart=SB['A'][8,:]+1,Pstart=SB['p'][8,:],izstart=SB['iz'][8,:]) 
 
change_earn  =((np.nanmean((SB_retro['w'][8:12,:]*p.wls[SB_retro['h'][8:12,:]]*adjustr[8:12,:]).sum(axis=0)))-\
                np.nanmean((SB['w'][8:12,:]      *p.wls[SB['h'][8:12,:]]      *adjustr[8:12,:]).sum(axis=0))) 
     

 
#Below annuitization like in Golosov (2024), assuming that agents sommth consumption 
change_pweal_s = (12-8)*((p.r/(1+p.r))*(1-(1/(1+p.r))**(p.T-8))**-1)#*np.mean((p.ρ*(SB['pb3']-SB['p'])*adjustr*(SB['ir']==1)).sum(axis=0)) 
 
 
# #Below annuitization like in Golosov (2024), NOT assuming that agents sommth consumption if PIH where r could be different than δ 
# change_pweal_s2 = share(p.r,p.δ,p.T,8,12-8).sum()#*np.mean((p.ρ*(SB['pb3']-SB['p'])*adjustr*(SB['ir']==1)).sum(axis=0)) 
 
# #Below model-consistent annuitization, where wealth is allocated according to consumption path. How to get it, 
# #use the intertemporal budget constraint and take out of summation c0 (future consumtion is replaced by ct/c0). 
# #Then manage the intertemporal BC to have c0=stuff: use it to get annuity value of future pension wealth. then 
# #sum the implied consumtion for periods 8 to 12. This can be checked against change_pweal_s2 
# ct_over_c0_discounted=np.mean(SB['c'][8:,:],axis=1)/np.mean(SB['c'][8,:])*adjustr[8:,0] 
# c0=np.mean((p.ρ*(SB['pb3']-SB['p'])*adjustr*(SB['ir']==1)).sum(axis=0))/ct_over_c0_discounted.sum() 
 
# change_pweal_d = c0*((np.mean(SB['c'][8:,:],axis=1)/np.mean(SB['c'][8,:]))[:4]).sum() 
 
#Finally, the marginal propensity to earn 
MPE = change_earn/(change_pweal_s) 
 
#Effect of pension reform on: 
     
#1) pension points 
eff_points=np.nanmean(np.diff(SP['p'][8:12,:],axis=0))-np.nanmean(np.diff(SB['p'][8:12,:],axis=0)) 

#2) behavioral pension points
eff_beh_points=np.nanmean(np.diff(SP['p'][8:12,:],axis=0))-np.nanmean(np.diff(SB['p'][8:12,:],axis=0))-np.mean(SP['pb'][8:12,:])
 
#3) full employment 
eff_earn=(np.nanmean(p.wls[SP['h'][8:12,:]]*SP['wh'][8:12,:])-np.nanmean(p.wls[SB['h'][8:12,:]]*SB['wh'][8:12,:]))*p.scale 
 
#4) earnings 
eff_emp=np.nanmean(SP['h'][8:12,:][SP['h'][8:12,:]>0]==3)-np.nanmean(SB['h'][8:12,:][SB['h'][8:12,:]>0]==3) 
 
#5) marginal work 
eff_marg=np.mean(SP['h'][8:12,:][SP['h'][8:12,:]>0]==1)-np.mean(SB['h'][8:12,:][SB['h'][8:12,:]>0]==1)
 
#6) earnings
eff_earn = (np.mean(SP['w'][8:12,:]*p.wls[SP['h'][8:12,:]]*(SP['h'][8:12,:]>1))-np.mean(SB['w'][8:12,:]*p.wls[SB['h'][8:12,:]]*(SB['h'][8:12,:]>1)))*p.scale
############################################ 
#Table with parameters 
###########################################      
table=r'\begin{table}[htbp]'+\
        r'\caption{Non-targeted moments}\label{table:nontargeted_moments}'+\
        r'\centering\footnotesize'+\
        r'\begin{tabular}{lcc}'+\
        r' \toprule '+\
        r" Effect of the reform on &   Data & Model  \\"+\
        r'\midrule   '+\
        r' Pension points   & 0.11 &'+p42(eff_points)+'\\\\'+\
        r' Behavioral pension points   & 0.05 &'+p42(eff_beh_points)+'\\\\'+\
        r' Work full time    & 0.03 &'+p42(eff_emp)+'\\\\'+\
        r' Marginal employment    & -0.07 &'+p42(eff_marg)+'\\\\'+\
        r' Non-marginal employment earnings (\euro)    & 1404 &'+p40(eff_earn)+'\\\\'+\
        r'\toprule   '+\
        r" Other moments &   Data & Model  \\"+\
        r'\midrule   '+\
        r' Marginal propensity to earn (MPE)      & -0.51\text{ to }-0.12 &'+p42(MPE)+'\\\\'+\
        r'  \bottomrule'+\
        r'\multicolumn{3}{l}{}'+\
      """\end{tabular}"""+\
      r'\end{table}' 
       
#Write table to tex file 
with open('C:/Users/32489/Dropbox/occupation/model/pfabio/output/table_nontargetd.tex', 'w') as f: 
    f.write(table) 
    f.close() 
     
