*Fabio playing around with existing GSOEP datasets*
clear all
use "C:\Users\32489\Dropbox\occupation\soepdata\data_sample_selection", replace
set seed 12345

* Create new variable hours_group with the following categories:
* == 1 if unemployed (pgemplst == 5 meaning not employed)
* == 2 if 0 < agreed upon hours <= 10
* == 3 if 10 < agreed upon hours <= 20
* == 4 if 20 < agreed upon hours <= 30
* == 5 if agreed upon hours is larger than 30


*Model start at age 29, when every women has a kid

gen adjust=1.0
*0.705


generate hours_group=. // a list of variables with only NA's
replace hours_group=1 if pgemplst == 5 
replace hours_group=2 if pgvebzeit > 0 & pgvebzeit <= 10 & pgvebzeit != .
replace hours_group=3 if pgvebzeit > 10 & pgvebzeit <= 20 & pgvebzeit != .
replace hours_group=4 if pgvebzeit > 20 & pgvebzeit <= 30 & pgvebzeit != .
replace hours_group=5 if pgvebzeit > 30 & pgvebzeit != .

*regular employed
gen irregular=0
replace irregular=1 if pgemplst==4


*Generate hours variables (the value makes sense to me)
gen hours=0 if pgemplst == 5 
replace hours= pgvebzeit if hours_group>=2 & hours_group<=5
sum hours if hours_group>=1 & hours_group<=5 &  treated==1

*Yearly earnings
gen earnings=pglabgro*12*adjust
replace earnings=0 if hours==0

*Yearly earnings without 
gen earnings_mod=earnings
replace earnings_mod=0 if pgemplst == 4

*Take adjust such that number below is 7694
sum earnings if sex==2 & syear>=1995 & syear<=2000  & treated==1 [weight= phrf]

sum hours if sex==2 & syear>=1995 & syear<=2000  & treated==1 [weight= phrf]

*Obtain hourly wages
gen log_wage=log(earnings/(hours*52))

gen employed=0
replace employed=1 if earnings>0

gen mortgage=0
replace mortgage=1 if hlf0087_h==1

********************************************************************************

probit employed age age_2 i.syear i.numberofchildren  i.education mortgage  if sex==2

gen sample=0
replace sample=1 if  e(sample)==1

*Get inverse of the mills ratio
predict gm 
replace gm=-gm
gen     lambda=normalden(gm)/(1-normal(gm))

reg log_wage age age_2 i.syear i.numberofchildren i.education lambda if sex==2 & sample==1

*Produce hidden productivity
predict ln_ly_ha, xb
gen shock=rnormal(0.0, e(rmse))

gen log_wage_corrected=log_wage
replace log_wage_corrected=ln_ly_ha+shock if employed==0

*Men's earnings
gen earnings_p=pglabgro_p*12*adjust
replace earnings_p=0 if pglabgro_p==0
reg earnings_p age age_2 i.syear 

********************************************************************************
keep if syear>=1995 & syear<=2002  & treated==1 & sex==2

*Create categories
pctile pct = log_wage_corrected [weight=phrf], nq(10) genp(percent) 
list pct in 1/10
xtile category = log_wage_corrected, cut(pct)
 
 
*Women's earnings
egen mean_logwage = wtmean(log_wage_corrected), weight(phrf) by(category) 
gen mean_wage = exp(mean_logwage)*17.04548*52
gen mean_wage2 = log(exp(mean_logwage)*17.04548*52)
tab mean_wage2

*Net worth
gen networth=w0111a*adjust
egen mean_networth = wtmean(networth), weight(phrf) by(category)
tab mean_networth



egen mean_earnings_p = wtmean(earnings_p), weight(phrf) by(category) 
tab mean_earnings_p