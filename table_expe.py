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
 
pτ = co.setup();pτ.tbase[3:11]=p.tbase[3:11]-0.175#0.22#197
Modτ = sol.solveEulerEquation(pτ,model='baseline') 
 
# pPN = co.setup();pPN.wls_point2=np.array([0.0,0.1,1.0,1.0]);pPN.standard_wls=False
# ModPN = sol.solveEulerEquation(pPN,model='baseline') 
 
pPN = co.setup();pPN.Pmax=1000000;pPN.add_points=1.68#1.4#35 
ModPN = sol.solveEulerEquation(pPN,model='pension reform') 

#pm = co.setup();pm.wls_point=0;pm.add_points=1.543 
#Modm = sol.solveEulerEquation(pm,model='pension reform') 
 
######################################## 
# simulate the models 
######################################## 
 
SB = sim.simNoUncer_interp(p,  ModB, Tstart=np.zeros(p.N,dtype=np.int16),Astart=p.startA,Pstart=np.ones((p.T,p.N))*p.startP,izstart=p.tw) 
#SB = sim.simNoUncer_interp(p,  ModB,Tstart=np.zeros(p.N,dtype=np.int16)+3,Astart=SB['A'],Pstart=SB['p'],izstart=SB['iz']) 
SP = sim.simNoUncer_interp(p,  ModP,Tstart=np.zeros(p.N,dtype=np.int16),Astart=p.startA,Pstart=np.ones((p.T,p.N))*p.startP,izstart=p.tw) 
Sτ = sim.simNoUncer_interp(pτ, Modτ,Tstart=np.zeros(p.N,dtype=np.int16),Astart=p.startA,Pstart=np.ones((p.T,p.N))*p.startP,izstart=p.tw) 
SPN= sim.simNoUncer_interp(pPN,ModPN,Tstart=np.zeros(p.N,dtype=np.int16),Astart=p.startA,Pstart=np.ones((p.T,p.N))*p.startP,izstart=p.tw) 
#Sm = sim.simNoUncer_interp(pm, Modm,Tstart=np.zeros(p.N,dtype=np.int16)+3,Astart=SB['A'],Pstart=SB['p']) 
 
 
######################################## 
# EXPERIMENTS 
######################################## 
 
 
adjustb=np.ones(SB['c'].shape)/((1+p.δ)**(np.cumsum(np.ones(p.T))-1.0))[:,None] 
t=0
EVP    =np.mean(((np.cumsum((adjustb*SP['v'])[::-1],axis=0)[::-1])*(1+p.δ)**t)[t]) 
EVτ    =np.mean(((np.cumsum((adjustb*Sτ['v'])[::-1],axis=0)[::-1])*(1+p.δ)**t)[t]) 
EVPN   =np.mean(((np.cumsum((adjustb*SPN['v'])[::-1],axis=0)[::-1])*(1+p.δ)**t)[t]) 
#EVm    =np.mean(((np.cumsum((adjustb*Sm['v'])[::-1],axis=0)[::-1])*(1+p.δ)**t)[t]) 
for i in np.linspace(1.00005,1.0299,100): 
     
    St= sim.simNoUncer_interp(p, ModB,cadjust=i,Tstart=np.zeros(p.N,dtype=np.int16),Astart=p.startA,Pstart=np.ones((p.T,p.N))*p.startP,izstart=p.tw) 
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
deficit_B=np.nansum(adjust*p.ρ*SB['p']*(SB['ir']==1))-np.nansum(adjust*SB['taxes'])#(np.nansum(expe_B*adjust[p.R:,:])  -np.nansum(tax_B*adjust[3:p.R,:])) 
deficit_P=np.nansum(adjust*p.ρ*SP['p']*(SP['ir']==1))-np.nansum(adjust*SP['taxes'])#(np.nansum(expe_P*adjust[p.R:,:])  -np.nansum(tax_P*adjust[3:p.R,:])) 
deficit_τ=np.nansum(adjust*pτ.ρ*Sτ['p']*(Sτ['ir']==1))-np.nansum(adjust*Sτ['taxes'])#(np.nansum(expe_τ*adjust[p.R:,:])  -np.nansum(tax_τ*adjust[3:p.R,:])) 
deficit_PN=np.nansum(adjust*pPN.ρ*SPN['p']*(SPN['ir']==1))-np.nansum(adjust*SPN['taxes'])#(np.nansum(expe_PN*adjust[p.R:,:])-np.nansum(tax_PN*adjust[3:p.R,:])) 

#3) Gender wage gaps in old age 
ggap_old_B=1.0-(np.nanmean(p.ρ*SB['p'][SB['ir']==1]))/np.nanmean(p.y_N[p.R,SB['iz']])
ggap_old_P=1.0-(np.nanmean(p.ρ*SP['p'][SP['ir']==1]))/np.nanmean(p.y_N[p.R,SB['iz']])
ggap_old_τ=1.0-(np.nanmean(pτ.ρ*Sτ['p'][Sτ['ir']==1]))/np.nanmean(pτ.y_N[p.R,SB['iz']])
ggap_old_PN=1.0-(np.nanmean(pPN.ρ*SPN['p'][SPN['ir']==1]))/np.nanmean(pPN.y_N[p.R,SB['iz']])
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
       r'\begin{tabular}{lcccc}'+\
       r' \toprule '+\
       r"& Pension &  Women's labor & Average age &  Welfare gains  \\"+\
       r"&gender gap &hours &  at retirement  & wrt baseline (\%)  \\"+\
       r'\midrule   '+\
       r' Baseline                                   &'+p43(ggap_old_B)  +'&'+p42(WLS_B)    +'&'+p42(30+ret_B) +'& 0.0\\\\'+\
       r' Caregiver credits                          &'+p43(ggap_old_P)  +'&'+p42(WLS_P)    +'&'+p42(30+ret_P) +'&'+p43(welf_P*100)+'\\\\'+\
       r' Caregiver credits, no threshold            &'+p43(ggap_old_PN) +'&'+p42(WLS_PN)  +'&'+p42(30+ret_PN) +'&'+p43(welf_PN*100)+'\\\\'+\
       r' Lower income taxes                         &'+p43(ggap_old_τ)  +'&'+p42(WLS_τ)    +'&'+p42(30+ret_τ) +'&'+p43(welf_τ*100)+'\\\\'+\
       r' \bottomrule'+\
       r'\end{tabular}'+\
       r'\begin{tablenotes}[flushleft]\small\item \textsc{Notes:} The experiments in the last three rows imply the same government deficit.'+\
       r' Welfare gains = increase in consumption at baseline to be indifferent with the experiment under analysis.'+\
       r' Reforms are in place while the child is 3-10 y.o.''\\\\'+\
       r'\end{tablenotes}'+\
      r'\end{threeparttable}'+\
      r'\end{table}' 
       
#Write table to tex file 
with open('C:/Users/32489/Dropbox/occupation/model/pfabio/output/table_expe.tex', 'w') as f: 
    f.write(table) 
    f.close() 
