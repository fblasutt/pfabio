import numpy as np
from numba import njit

# consav
from consav import linear_interp # for linear interpolation


@njit
def index_func(i_n,i_m,Nn,Nm):
    return i_n*Nm + i_m

@njit
def compute(out_c,out_d,out_v,holes,
            m,n,c,d,
            num,
            w,
            gamma_c,maxHours,gamma_h,rho,agrid,pgrid,beta,r,wt,tau,y_N,E_bar_now,delta,valt=np.array([[]])):
    
    # a. infer shape
    Nb,Na = w.shape
        
    # b. indicator for valid and interesting choice or not
    valid = np.ones((Nb,Na),dtype=np.bool_)
    for i_b in range(Nb):
        for i_a in range(Na):

            valid[i_b,i_a] &= (np.imag(c[i_b,i_a]) == 0)
            valid[i_b,i_a] &= (np.imag(d[i_b,i_a]) == 0)
            valid[i_b,i_a] &= (~np.isnan(w[i_b,i_a]))
            valid[i_b,i_a] &= c[i_b,i_a] >= -0.50
            valid[i_b,i_a] &= d[i_b,i_a] >= -0.50
            valid[i_b,i_a] &= m[i_b,i_a] > -0.1
            valid[i_b,i_a] &= n[i_b,i_a] > -0.1
            #valid[i_b,i_a] &= m[i_b,i_a] < par.m_max + 1
            #valid[i_b,i_a] &= n[i_b,i_a] < par.n_max + 1

            if valt.size > 0:
                valid[i_b,i_a] &= w[i_b,i_a] > valt[i_b,i_a]

    # c. upper envelope
    out_c[:,:] = np.nan
    out_d[:,:] = np.nan
    out_v[:,:] = -np.inf

    if valid.sum() >= 100:
        
        # i. allocate holes
        #holes = np.ones((par.Nn,par.Nm))
        #holes = np.ones((Nb,Na))

        # ii. upperenvelope
        for i_b in range(Nb):
            for i_a in range(Na):
                for tri in range(2):            
                    upperenvelope(out_c,out_d,out_v,holes,i_a,i_b,tri,
                                  m,n,c,d,
                                  Na,Nb,valid,num,w,
                                  gamma_c,maxHours,gamma_h,rho,agrid,pgrid,beta,r,wt,tau,y_N,E_bar_now,delta)
                    
                    upperenvelope(out_c,out_d,out_v,holes,i_a,i_b,tri,
                                  m,n,c,d,
                                  Na,Nb,valid,num,w,
                                  gamma_c,maxHours,gamma_h,rho,agrid,pgrid,beta,r,wt,tau,y_N,E_bar_now,delta,egm_extrap_w=-0.5)
                    
                    
        # iii. fill holes
        fill_holes(out_c,out_d,out_v,holes,w,num,gamma_c,maxHours,gamma_h,rho,agrid,pgrid,beta,r,wt,tau,y_N,E_bar_now,delta,Nb,Na)

@njit
def upperenvelope(out_c,out_d,out_v,holes,i_a,i_b,tri,m,n,c,d,Na,Nb,valid,num,w,
                  gamma_c,maxHours,gamma_h,rho,agrid,pgrid,beta,r,wt,tau,y_N,E_bar_now,delta,
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
    
    if ~valid[i_b_1,i_a_1] or ~valid[i_b_2,i_a_2] or ~valid[i_b_3,i_a_3]:
        return
   
    # b. simplex in (m,n)-space
    m1 = m[i_b_1,i_a_1]
    m2 = m[i_b_2,i_a_2]
    m3 = m[i_b_3,i_a_3]

    n1 = n[i_b_1,i_a_1]
    n2 = n[i_b_2,i_a_2]
    n3 = n[i_b_3,i_a_3]

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

    # e. loop through common grid nodes in interior of bounding box
    for i_n in range(in_low,in_high):
        for i_m in range(im_low,im_high):

            if holes[i_n,i_m]>0:
                # i. common grid values
                m_now = pgrid[i_m]
                n_now = agrid[i_n]
    
                # ii. barycentric coordinates
                w1 = ((n2-n3)*(m_now-m3) + (m3-m2)*(n_now-n3)) / denom
                w2 = ((n3-n1)*(m_now-m3) + (m1-m3)*(n_now-n3)) / denom
                w3 = 1 - w1 - w2
    
                # iii. exit if too much outside simplex
                if w1 < egm_extrap_w or w2 < egm_extrap_w or w3 < egm_extrap_w:
                      continue
    
                # iv. interpolate choices
                if num == 1: # ucon, interpolate c and d
    
                    c_interp = w1*c[i_b_1,i_a_1] + w2*c[i_b_2,i_a_2] + w3*c[i_b_3,i_a_3]
                    d_interp = w1*d[i_b_1,i_a_1] + w2*d[i_b_2,i_a_2] + w3*d[i_b_3,i_a_3]
                    a_interp = m_now + d_interp*wt/E_bar_now #m_now - d_interp*wt/E_bar_now                     #points
                    b_interp = n_now*(1+r) - c_interp + wt*(1-tau)*d_interp + y_N#(n_now + c_interp - wt*(1-tau)*d_interp - y_N)/(1+r) #assets
    
                elif num == 2: # dcon, interpolate c
    
                    c_interp = w1*c[i_b_1,i_a_1] + w2*c[i_b_2,i_a_2] + w3*c[i_b_3,i_a_3]
                    d_interp = 0.0
                    a_interp = m_now#m_now 
                    b_interp = n_now*(1+r) - c_interp + y_N#(n_now + c_interp - y_N)/(1+r) #assets
    
                elif num == 3: # acon, interpolate d
    
                    d_interp = w1*d[i_b_1,i_a_1] + w2*d[i_b_2,i_a_2] + w3*d[i_b_3,i_a_3]
                    a_interp = m_now + d_interp*wt/E_bar_now 
                    b_interp = 0.0
                    c_interp = n_now*(1+r)+wt*(1-tau)*d_interp + y_N#n_now+wt*(1-tau)*d_interp + y_N
                    
    
                if c_interp <= 0.0 or d_interp < 0.0 or a_interp < 0 or b_interp < 0:
                    continue
    
                # v. value-of-choice
                #w_interp = linear_interp.interp_2d(par.grid_b_pd,par.grid_a_pd,w,b_interp,a_interp)
                #v_interp = utility.func(c_interp,par) + w_interp
    
                w_interp = linear_interp.interp_2d(agrid,pgrid,w,b_interp,a_interp)
                v_interp=c_interp**(1-gamma_c)/(1-gamma_c)+\
                     beta*(maxHours - d_interp)**(1 - gamma_h) / (1 - gamma_h)+\
                         1/(1+delta)*w_interp
                # vi. update if max
                if v_interp >out_v[i_n,i_m]:
    
                    out_v[i_n,i_m] = v_interp
                    out_c[i_n,i_m] = c_interp
                    out_d[i_n,i_m] = d_interp
                    holes[i_n,i_m] = 0

@njit
def fill_holes(out_c,out_d,out_v,holes,w,num,gamma_c,maxHours,gamma_h,rho,agrid,pgrid,beta,r,wt,tau,y_N,E_bar_now,delta,Nn,Nm):

    # a. locate global bounding box with content
    print(np.mean(holes))
    i_n_min = 0
    i_n_max = Nn-1
    min_n = np.inf
    max_n = -np.inf

    i_m_min = 0
    i_m_max = Nm-1
    min_m = np.inf
    max_m = -np.inf

    for i_n in range(Nn):
        for i_m in range(Nm):#?????

            m_now = pgrid[i_m]
            n_now = agrid[i_n]

            if holes[i_n,i_m] == 1: continue

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
    i_n_min=0
    i_m_min=0
    for i_n in range(i_n_min,i_n_max):
        for i_m in range(i_m_min,i_m_max):
            
            if holes[i_n,i_m] == 0: # if not hole
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

                    if holes[i_n_close,i_m_close] == 1: # if itself a hole
                        continue

                    if num == 1: # ucon, interpolate c and d

                        c_interp = out_c[i_n_close,i_m_close]
                        d_interp = out_d[i_n_close,i_m_close]
                        a_interp = m_now + d_interp*wt/E_bar_now #m_now - d_interp*wt/E_bar_now                     #points
                        b_interp = n_now*(1+r) - c_interp + wt*(1-tau)*d_interp + y_N#(n_now + c_interp - wt*(1-tau)*d_interp - y_N)/(1+r) #assets

                    elif num == 2: # dcon, interpolate c

                        c_interp = out_c[i_n_close,i_m_close]
                        d_interp = 0.0
                        a_interp = m_now#m_now 
                        b_interp = n_now*(1+r) - c_interp + y_N#(n_now + c_interp - y_N)/(1+r) #assets

                    elif num == 3: # acon, interpolate d

                        d_interp = out_d[i_n_close,i_m_close]
                        a_interp = m_now + d_interp*wt/E_bar_now 
                        b_interp = 0.0
                        c_interp = n_now*(1+r)+wt*(1-tau)*d_interp + y_N#n_now+wt*(1-tau)*d_interp + y_N
                        

                    if c_interp <= 0.0 or d_interp < 0.0 or a_interp < 0 or b_interp < 0:
                        continue

                    # value-of-choice
                    #w_interp = linear_interp.interp_2d(par.grid_b_pd,par.grid_a_pd,w,b_interp,a_interp)
                    #v_interp = utility.func(c_interp,par) + w_interp
                    
                    w_interp = linear_interp.interp_2d(agrid,pgrid,w,b_interp,a_interp)
                    v_interp=c_interp**(1-gamma_c)/(1-gamma_c)+\
                     beta*(maxHours - d_interp)**(1 - gamma_h) / (1 - gamma_h)+\
                       1/(1+delta)*w_interp

                    # update if better
                    if v_interp > out_v[i_n,i_m]:

                        out_v[i_n,i_m] = v_interp
                        out_c[i_n,i_m] = c_interp
                        out_d[i_n,i_m] = d_interp
    