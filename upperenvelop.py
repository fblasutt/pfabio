import numpy as np
from numba import njit,prange

###############################################################################
# This is an adaptation of the CONVSAV packages
# https://github.com/NumEconCopenhagen/ConsumptionSavingNotebooks/tree/master/03.%20G2EGM
# related to DruedahlThomas and Jørgensen (2017).
# It is a way to combine EGM with upper envelop to solve problems with two
# continuous grids. It uses triangularization for the 2-way interpolation
###############################################################################
 
from consav import linear_interp # for linear interpolation

@njit
def index_func(i_n,i_m,Nn,Nm):
    return i_n*Nm + i_m

@njit
#@profile
def compute(out_c,out_n,pmua,out_v,holes,
            m,n,c,m_bc,n_bc,c_bc,
            num,
            w,
            γc,maxHours,γh,ρ,agrid,pgrid,β,r,wt,τ,y_N,E_bar_now,δ,q,amin,wls,mp,valt=np.array([[]])):
    
    # a. infer shape
    Nb,Na,nw = w.shape
 
    # b. indicator for valid and interesting choice or not
    valid = np.ones((Nb,Na,nw),dtype=np.bool_)
    for i_b in range(Nb):
        for i_a in range(Na):
            for i_w in range(nw):

                #Non-interesting choices below
                valid[i_b,i_a,i_w] &= (np.imag(c[i_b,i_a,i_w]) == 0)
                valid[i_b,i_a,i_w] &= (~np.isnan(w[i_b,i_a,i_w]))
                valid[i_b,i_a,i_w] &= c[i_b,i_a,i_w] >= 0.0
                #valid[i_b,i_a] &= m[i_b,i_a] > -0.1
                #valid[i_b,i_a] &= n[i_b,i_a] > -0.1
                #valid[i_b,i_a] &= m[i_b,i_a] < par.m_max + 1
                #valid[i_b,i_a] &= n[i_b,i_a] < par.n_max + 1
    
                # if valt.size > 0:
                #     valid[i_b,i_a,i_w] &= w[i_b,i_a,i_w] > valt[i_b,i_a,i_w]

    # c. upper envelope
    out_c[:,:,:] = np.nan
    out_n[:,:,:] = np.nan
    pmua[:,:,:] = np.nan
    out_v[:,:,:] = -np.inf

    if valid.sum() >= 0:
        
        # i. allocate holes
        #holes = np.ones((par.Nn,par.Nm))
        #holes = np.ones((Nb,Na))

        # ii. upperenvelope
        for i_b in prange(Nb):
            for i_a in prange(Na):
                for i_w in prange(nw):
                    for tri in prange(2): #consider both upper and lower triangle     
                    
          
                        upperenvelope(out_c,out_n,pmua,out_v,holes,i_a,i_b,tri,i_w,
                                      m,n,c,m_bc,n_bc,c_bc,
                                      Na,Nb,valid,num,w,
                                      γc,maxHours,γh,ρ,agrid,pgrid,β,r,wt,τ,y_N,E_bar_now,δ,q,amin,wls,mp)                    
                        
        # iii. fill holes (technique: nearest neighbor)
        fill_holes(out_c,out_n,pmua,out_v,holes,w,num,γc,maxHours,γh,ρ,agrid,pgrid,β,r,wt,τ,y_N,E_bar_now,δ,q,amin,wls,mp,Nb,Na,nw)

@njit
#@profile
def upperenvelope(out_c,out_n,pmua,out_v,holes,i_a,i_b,tri,i_w,
                  m_ok,n_ok,c_ok,m_bc,n_bc,c_bc,
                  Na,Nb,valid,num,w,
                  γc,maxHours,γh,ρ,agrid,pgrid,β,r,wt,τ,y_N,E_bar_now,δ,q,amin,wls,mp,
                  egm_extrap_add=2,egm_extrap_w=-0.25):
    
    # a. simplex in (a,b)-space (or similar with constrained choices)
    i_b_1 = i_b
    i_a_1 = i_a

    if i_b == Nb-1: return
    i_b_2 = i_b+1
    i_a_2 = i_a

    i_b_3 = -1 # to be overwritten
    i_a_3 = -1 # to be overwritten

    if tri == 0:
        if i_a == 0 or i_b == Nb-1: return
        i_b_3 = i_b+1
        i_a_3 = i_a-1
    else:
        if i_a == Na-1: return
        i_b_3 = i_b
        i_a_3 = i_a+1
    
    if ~valid[i_b_1,i_a_1,i_w] or ~valid[i_b_2,i_a_2,i_w] or ~valid[i_b_3,i_a_3,i_w]:
        return
   
    #Create a loop where the first entry () is not constrained and the second
    #entry is constrained (1)
    
    for j in range(2):
        
        #Consider j specific case:
        if j==0:m=m_ok.copy();n=n_ok.copy();c=c_ok.copy()
        if j==1:m=m_bc.copy();n=n_bc.copy();c=c_bc.copy()
        
        #Not constrained
        m1 = m[i_b_1,i_a_1,i_w]
        m2 = m[i_b_2,i_a_2,i_w]
        m3 = m[i_b_3,i_a_3,i_w]
    
        n1 = n[i_b_1,i_a_1,i_w]
        n2 = n[i_b_2,i_a_2,i_w]
        n3 = n[i_b_3,i_a_3,i_w]
    
        # c. boundary box values and indices in common grid
        m_max = np.fmax(m1,np.fmax(m2,m3))
        m_min = np.fmin(m1,np.fmin(m2,m3))
        n_max = np.fmax(n1,np.fmax(n2,n3))
        n_min = np.fmin(n1,np.fmin(n2,n3))
    
        im_low = 0
        if m_min >= 0: im_low = linear_interp.binary_search(0,Na,pgrid,m_min)
        im_high = linear_interp.binary_search(0,Na,pgrid,m_max) + 1
        
        in_low = 0
        if n_min >= 0: in_low = linear_interp.binary_search(0,Nb,agrid,n_min)
        in_high = linear_interp.binary_search(0,Nb,agrid,n_max) + 1
        
        # correction to allow for more extrapolation
        im_low = np.fmax(im_low-egm_extrap_add,0)
        im_high = np.fmin(im_high+egm_extrap_add,Na)
        in_low = np.fmax(in_low-egm_extrap_add,0)
        in_high = np.fmin(in_high+egm_extrap_add,Nb)
    
        # d. prepare barycentric interpolation
        denom = (n2-n3)*(m1-m3)+(m3-m2)*(n1-n3)
        dn23=n2-n3
        dn31=n3-n1
        dm32=m3-m2
        dm13=m1-m3
    
    
        # e. loop through common grid nodes in interior of bounding box
        for i_n in prange(in_low,in_high):       
            
            n_now = agrid[i_n]
            den=n_now-n3
            for i_m in prange(im_low,im_high):
                if holes[i_n,i_m,i_w,j]>0:
                    
                    # i. common grid values
                    m_now = pgrid[i_m]
                    dem=m_now-m3
                                  
                    # ii. barycentric coordinates
                    w1 = (dn23*dem + dm32*den) / denom
                    w2 = (dn31*dem + dm13*den) / denom
                    w3 = 1 - w1 - w2
                    
                    # iii. exit if too much outside simplex
                    if min(w1,w2,w3)<egm_extrap_w:continue
                          
        
                   
                    # iv. interpolate choices (num inicates wls) 
    
                    
                    if j==0:
                        #No borrowing constrained case
                        c_interp = w1*c[i_b_1,i_a_1,i_w] + w2*c[i_b_2,i_a_2,i_w] + w3*c[i_b_3,i_a_3,i_w]
                        a_interp = m_now + np.maximum(np.minimum(mp*wls*wt[i_w]/E_bar_now,1.0),wls*wt[i_w]/E_bar_now)                  #points
                        b_interp = n_now*(1+r) - c_interp + wt[i_w]*(1-τ)*wls + y_N[i_n,i_m,i_w]
    
                    
                    if j==1:
                        #Borrowing constrained case
                        a_interp = m_now + np.maximum(np.minimum(mp*wls*wt[i_w]/E_bar_now,1.0),wls*wt[i_w]/E_bar_now)  
                        b_interp = amin
                        c_interp = n_now*(1+r)+wt[i_w]*(1-τ)*wls + y_N[i_n,i_m,i_w]
                 
        
                    if c_interp <= 0.0 or a_interp < 0 or b_interp < amin:
                        continue
                    
                    # v. value of choice
                    w_interp = linear_interp.interp_2d(agrid,pgrid,w[:,:,i_w],b_interp,a_interp)
                    v_interp=np.log(c_interp)-β*wls**(1+1/γh) / (1+1/γh)-q+w_interp/(1+δ)     
    
                    # vi. update if max
                    if v_interp >out_v[i_n,i_m,i_w]:
        
                        out_v[i_n,i_m,i_w] = v_interp
                        out_c[i_n,i_m,i_w] = c_interp
                        out_n[i_n,i_m,i_w] = b_interp
                        pmua[i_n,i_m,i_w]  = a_interp
                        holes[i_n,i_m,i_w,j] = 0

@njit
def fill_holes(out_c,out_n,pmua,out_v,holes,w,num,γc,maxHours,γh,ρ,agrid,pgrid,β,r,wt,τ,y_N,E_bar_now,δ,q,amin,wls,mp,Nn,Nm,nw):

    # a. locate global bounding box with content
    i_n_min = 0
    i_n_max = Nn-1
    min_n = np.inf
    max_n = -np.inf

    i_m_min = 0
    i_m_max = Nm-1
    min_m = np.inf
    max_m = -np.inf

    for i_n in range(Nn):
        for i_m in range(Nm):
            for i_w in range(nw):
                for j in range(2):

                    
                    m_now = pgrid[i_m]
                    n_now = agrid[i_n]
        
                    if holes[i_n,i_m,i_w,j] == 1: continue
        
                    if m_now < min_m:
                        min_m = m_now
                        i_m_min = i_m
        
                    if m_now > max_m:
                        max_m = m_now
                        i_m_max = i_m
        
                    if n_now < min_n:
                        min_n = n_now
                        i_n_min = i_n
                    
                    if n_now > max_n:
                        max_n = n_now
                        i_n_max = i_n

    # b. loop through m, n, k nodes to detect holes
    i_n_max = np.fmin(i_n_max+1,Nn)
    i_m_max = np.fmin(i_m_max+1,Nm)
    for i_n in range(i_n_min,i_n_max):
        for i_m in range(i_m_min,i_m_max):
            for i_w in range(nw):
                for j in range(2):
                    if holes[i_n,i_m,i_w,j] == 0: # if not hole
                        continue
         
                    m_now = pgrid[i_m]
                    n_now = agrid[i_n]
                    m_add = 2
                    n_add = 2
                
                    # loop over points close by
                    i_n_close_min = np.fmax(0,i_n-n_add)
                    i_n_close_max = np.fmin(i_n+n_add+1,Nn)
        
                    i_m_close_min = np.fmax(0,i_m-m_add)
                    i_m_close_max = np.fmin(i_m+m_add+1,Nm)
        
                    for i_n_close in range(i_n_close_min,i_n_close_max):
                        for i_m_close in range(i_m_close_min,i_m_close_max):
                            for j in range(2):#= is not bc, 1 is bc
            
                                if holes[i_n_close,i_m_close,i_w,j] == 1: # if itself a hole
                                    continue
            
                                          
                                    #No borrowing constrained case
                                    if j==0:
                                        c_interp = out_c[i_n_close,i_m_close,i_w]
                                        a_interp = m_now + np.maximum(np.minimum(mp*wls*wt[i_w]/E_bar_now,1.0),wls*wt[i_w]/E_bar_now)                    #points
                                        b_interp = n_now*(1+r) - c_interp + wt[i_w]*(1-τ)*wls + y_N[i_n,i_m,i_w]
                 
                                     
                                                                     
                                   #Borrowing constrained case
                                    if j==1:
                                        a_interp = m_now + np.maximum(np.minimum(mp*wls*wt[i_w]/E_bar_now,1.0),wls*wt[i_w]/E_bar_now)  
                                        b_interp = amin
                                        c_interp = n_now*(1+r)+wt[i_w]*(1-τ)*wls + y_N[i_n,i_m,i_w]
                                       
                                        
                                    #Value of choice
                                    w_interp = linear_interp.interp_2d(agrid,pgrid,w[:,:,i_w],b_interp,a_interp)
                                    v_interp=np.log(c_interp)-β*wls**(1+1/γh) / (1+1/γh)-q+1/(1+δ)*w_interp
                                                
                                       
                                    if c_interp <= 0.0 or a_interp < 0 or b_interp < amin:
                                       continue
                    
                                   # vi. update if max
                                    if v_interp >out_v[i_n,i_m,i_w]:
                       
                                        out_v[i_n,i_m,i_w] = v_interp
                                        out_c[i_n,i_m,i_w] = c_interp
                                        out_n[i_n,i_m,i_w] = b_interp
                                        pmua[i_n,i_m,i_w]  = a_interp
                  