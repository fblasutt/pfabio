# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 08:41:19 2024

@author: 32489
"""


import numpy as np
import scipy
import pickle
import dfols

def TikTak(f,xl,xu,N,share,skip_first_step=False):
    
    """
    f : function to be minimized, should return a float number and shold have a list as input
    xl: list, lower bound of parameters to consider
    pl: list, uppe bound of parameters to consider
    N:  if N=x, we will draw 2**N sobol points
    share: share of the 2**N best sobol points to consider for the 2nd step of TikTak
    
    
    
    """
    
    # Set the seed
    
    
    #Generate sobol sequence
    NP=len(xl)
    sampler = scipy.stats.qmc.Sobol(d=NP, scramble=False)
    sample_notscaled = sampler.random_base2(m=N)
    sample= scipy.stats.qmc.scale(sample_notscaled, xl,xu)
    
    
    if skip_first_step:
        
        #load the file for later
        with open('first_step.pkl', 'rb') as file: first_step=pickle.load(file) 
        
    else:
        
        #Evaluate all the sobol points and store point + fit in array first_step
        first_step_params = np.zeros((2**N,NP))
        first_step_fit    = np.zeros(2**N)
        
        
        for i in range(2**N):
            
            first_step_params[i] = sample[i]
            first_step_fit[i]   =  (np.array(f(first_step_params[i]))**2).sum()
        
        #sort the two arrays and store them in list "first_step", then pickle the file
        order=np.argsort(first_step_fit)   
        first_step = [first_step_fit[order],first_step_params[order]]
        
        #store the file for later
        with open('first_step.pkl', 'wb+') as file: pickle.dump(first_step,file) 
    
    
    #determine how many points to consider for minimization step
    N2 = int(share*2**N)
    
    second_step_params = np.zeros((N2,NP))
    second_step_fit    = np.zeros(N2)+1e10
    
    best = first_step[1][0]
    
    for i in range(N2):
        
        #get the point to consider
        θ = min(max(0.1,(i/N2)**0.5),0.995)
        point = (1-θ)*first_step[1][i]+θ*best
       
        #store the point
        second_step_params[i]=point+0.0
        
        #evalueate the point
        #res = scipy.optimize.minimize(f,point,bounds=list(zip(list(xl), list(xu))),method='Nelder-Mead',tol=1e-5)
        
        #Optimization below
        res=dfols.solve(f, point, rhobeg = 0.3, rhoend=1e-5, maxfun=200, bounds=(np.array(xl),np.array(xu)),
                        npt=len(point)+5,scaling_within_bounds=True, 
                        user_params={'tr_radius.gamma_dec':0.98,'tr_radius.gamma_inc':1.0,
                                      'tr_radius.alpha1':0.9,'tr_radius.alpha2':0.95},
                        objfun_has_noise=False)
        
        second_step_fit[i]=(np.array(res.f)**2).sum()
        second_step_params[i]=res.x
        
        #find the best point so far
        order=np.argsort(second_step_fit)
        second_step = [second_step_fit[order],second_step_params[order]]
        best = second_step[1][0]
        

    return sample,first_step,second_step


# def rastrigin(x):
#     return np.sum(x*x - 10*np.cos(2*np.pi*x)) + 10*np.size(x)  

# lower_bounds = np.repeat(-5.12,2)  # lower bounds on each dimension
# upper_bounds = np.repeat( 5.12,2)  # upper bounds on each dimension

# sample, first_step, second_step = TikTak(rastrigin,lower_bounds,upper_bounds,13,0.01)