*Fabio playing around with existing GSOEP datasets*
clear all
use "C:\Users\32489\Dropbox\occupation\soepdata\data_sample_selection", replace
set seed 12345
keep if sex==2
replace phrf=int(phrf)
keep if age>=20 & age<65 
*keep if east_germany==0
* Create new variable hours_group with the following categories:
* == 1 if unemployed (pgemplst == 5 meaning not employed)
* == 2 if 0 < agreed upon hours <= 10
* == 3 if 10 < agreed upon hours <= 20
* == 4 if 20 < agreed upon hours <= 30
* == 5 if agreed upon hours is larger than 30


**# Bookmark #1
*Model start at age 29, when every women has a kid

gen adjust=15819.30/20317
*15819.30/18787.67
*7694/12582.09
*15819.30/19029.33
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

gen mortgage=3
replace mortgage=0 if hlf0087_h==2
replace mortgage=1 if hlf0087_h==1

********************************************************************************
* Women's pension points
********************************************************************************
gen points = 0
replace points =  min((earnings / 27740.65230618203),2) if irregular==0


********************************************************************************
* Women here
********************************************************************************

probit employed age age_2 i.syear  i.mortgage

gen sample=0
replace sample=1 if  e(sample)==1

*Get inverse of the mills ratio
predict gm 
replace gm=-gm
gen     lambda=normalden(gm)/(1-normal(gm))

reg log_wage age age_2 i.syear lambda  [weight= phrf ]
predict Wdelta,resid

gen Wcons =_b[_cons]+_b[2000.syear]
gen Wage  =_b[age]
gen Wage2 =_b[age_2]

*Produce hidden productivity
bysort pid (syear): gen d_log_wage=log_wage-log_wage[_n-1] if syear==syear[_n-1]+1
reg d_log_wage i.age i.syear i.numberofchildren i.education [weight=phrf] if sex==2 


*Get residuals and predition
predict y_w if e(sample)==1, resid


gen u_0=y_w
bys pid: gen  u_1= y_w[_n-1] if syear==syear[_n-1]+1  
bys pid: gen  u_2= y_w[_n-2] if syear==syear[_n-2]+2 
bys pid: gen  u_3= y_w[_n-3] if syear==syear[_n-3]+3  
bys pid: gen  u_4= y_w[_n-4] if syear==syear[_n-4]+4  
bys pid: gen  u_5= y_w[_n-5] if syear==syear[_n-5]+5 

gen expect=u_2*(u_0+u_1+u_2+u_3+u_4)  

sum expect [weight=phrf]

g var_pers=r(mean)

egen wvar=sd(log_wage) if sex==2,by(age)
replace wvar=wvar^2
binscatter wvar age if age>=30 & age<=60 

nlsur (wvar={sigma02}+(age-30)*var_pers) [weight=phrf] if age>=30 & age<=35 & var_pers!=.
gen sigma02=_b[/sigma02]


********************************************************************************
* Men here
********************************************************************************


*Men's earnings
gen earnings_p=pglabgro_p*12*adjust
replace earnings_p=. if pglabgro_p==0

gen log_earnings_p=log(earnings_p)
reg log_earnings_p age age_2 i.syear [weight=phrf]

gen Mcons =_b[_cons]+_b[2000.syear]
gen Mage  =_b[age]
gen Mage2 =_b[age_2]
predict Mdelta,resid


bysort pid (syear): gen d_log_earnings_p=log_earnings_p-log_earnings_p[_n-1] if syear==syear[_n-1]+1
reg d_log_earnings_p i.age i.syear i.numberofchildren i.education [weight=phrf]  if sex==2 & age>=23 & age<65 


*Get residuals and predition
predict y_m if e(sample)==1, resid


gen u_0m=y_m
bys pid: gen  u_1m= y_m[_n-1] if syear==syear[_n-1]+1  
bys pid: gen  u_2m= y_m[_n-2] if syear==syear[_n-2]+2 
bys pid: gen  u_3m= y_m[_n-3] if syear==syear[_n-3]+3  
bys pid: gen  u_4m= y_m[_n-4] if syear==syear[_n-4]+4  
bys pid: gen  u_5m= y_m[_n-5] if syear==syear[_n-5]+5 

gen expectm=u_2m*(u_0m+u_1m+u_2m+u_3m+u_4m)  

sum expectm [weight=phrf]

g var_persm=r(mean)

egen wvarm=sd(log_earnings_p) if sex==2,by(age)
replace wvarm=wvarm^2
binscatter wvarm age if age>=30 & age<=60 

nlsur (wvarm={sigmam02}+(age-30)*var_persm) [weight=phrf]  if age>=30 & age<=35 & var_persm!=.
gen sigmam02=_b[/sigmam02]

********************************************************************************
*Store results
file open myfile using "C:\Users\32489\Dropbox\occupation\model\pfabio\earnings_est.csv", write replace

file write myfile "Men:" _n

foreach v of varlist Mcons Mage Mage2 sigmam02 expectm {
       sum `v' [weight=phrf]
	   local vmean: display %10.8fc `r(mean)'
	   file write myfile "`vmean' " _n
}

file write myfile "Women:" _n

foreach v of varlist Wcons Wage Wage2 sigma02 expect {
       sum `v' [weight=phrf]
	   local vmean: display %10.8fc `r(mean)'
	   file write myfile "`vmean' " _n
}

file close myfile
 

********************************************************************************
*Average earnings
gen age_3=age^3
reg earnings_mod age age_2  i.syear [weight=phrf] if  earnings_mod>0
replace pred35 = _b[_cons]+_b[2000.syear]+_b[age]*35+_b[age_2]*35^2
*+_b[age_3]*35^3
sum pred35

*reg earnings_mod i.age  i.syear [weight=phrf] if age>=30 & earnings_mod>0
*& earnings>0  
*replace pred35 = _b[_cons]+_b[2000.syear]+_b[35.age]
*+_b[age_3]*35^3
*sum pred35
********************************************************************************

*keep if syear>=1995 & syear<=2002  & treated==1 & sex==2

*Create categories, number should be the same as the Python code
*gen Wdelta=log_wage-log_wage_hat
*gen Mdelta=log_earnings_p-log_earnings_p_hat

keep if Wdelta!=. & Mdelta!=.


_pctile Wdelta [weight=phrf], p(25 75)
gen pctw=r(r1) if _n==1
replace pctw=r(r2) if _n==2
xtile categoryw = Wdelta, cut(pctw)

_pctile Mdelta [weight=phrf], p(25 75)
gen pctm=r(r1) if _n==1
replace pctm=r(r2) if _n==2
xtile categorym = Mdelta, cut(pctm)



*Generate overall categories
gen category=.
replace category = (categoryw-1)*3+(categorym-1)


* Generate frequencies
egen summ_weights=total( phrf )
gen weight= phrf /summ_weights
egen _frequency=total( weight),by(category)

tab _frequency

*Net worth
gen networth=w0111a*adjust

/*
egen _networth = wtmean(networth), weight(phrf) by(category)
tab _networth
*/

gen nw_hat=0
gen ta1=.
gen ta2=.
gen acons=.

foreach i of numlist 0/8{
reg networth age age_2 [weight=phrf] if sex==2 & category==`i'
replace acons=_b[_cons]
replace  ta1=_b[age]
replace  ta2=_b[age_2]


	replace nw_hat=acons+ta1*29+ta2*29^2 if sex==2 & category==`i'
	

}

egen _networth = wtmean(nw_hat), weight(phrf) by(category)
tab _networth


*Pension points
gen points_hat=0
gen tp1=.
gen tp2=.
gen pcons=.

foreach i of numlist 0/8{
reg points age age_2 [weight=phrf] if sex==2 & category==`i'
replace pcons=_b[_cons]
replace  tp1=_b[age]
replace  tp2=_b[age_2]

foreach a of numlist 18/28{
	
	replace points_hat=points_hat+pcons+tp1*`a'+tp2*`a'^2 if sex==2 & category==`i'
}
	

}

egen _points = wtmean(points_hat), weight(phrf) by(category)
tab _points

*Average earnings
sum earnings [weight= phrf ]

*Keep only what needed
bysort category: keep if _n==1

keep _*

*Save
export delimited "C:\Users\32489\Dropbox\occupation\model\pfabio\categories.csv"
