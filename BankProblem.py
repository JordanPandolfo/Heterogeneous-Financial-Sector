#----------------------#
#                      # 
#   Import Packages    #
#                      # 
#----------------------#
import numpy as np
import itertools
import matplotlib.pyplot as plt
import time
from scipy.interpolate import RegularGridInterpolator

#-----------------#
#                 #
#   Parameters    #
#                 #
#-----------------#
p = np.array((.5,.5))

aa, ab, alen = .5, 20, 100
da,db,dlen   = 0,200, 300
la,lb,llen   = 0.1,200, 300
ta,tb,tlen   = 1e-02,.06,100
    
arange = np.arange(aa,ab,(ab-aa)/alen)
drange = np.arange(da,db,(db-da)/dlen)
lrange = np.arange(la,lb,(lb-la)/llen)
trange = np.arange(ta,tb,(tb-ta)/tlen)

erange = np.array((.1,-.1))
ebar = 0.1

beta = .97
alpha = 1
Rd = (1/beta-1)*(1-.35)+1
Q = beta*(p[0]*(1+erange[0])+p[1]*(1+erange[1]))*.92
delta_theta = -0.05
H = 0

#----------------------#
#                      # 
#   Model Functions    #
#                      # 
#----------------------#
def g(L,theta):
    return alpha*(L*theta)**2/2

def u(c):
    return np.log(c)

a1 = .01
a2 = .1
a3 = 30

#def phi(lp):
#    return a1/(1+np.exp(a2*(lp-a3)))

def phi(lp):
    if lp < 60:
        return .005-(.005/60)*lp
    if lp >= 60:
        return 0
       
#------------------------------------------------------------------#
#                                                                  #
#   Pre-compute indices/grid values that satisfy non-negativity    #
#      and equity constraints.  Pre-compute utility values         #
#                                                                  #  
#------------------------------------------------------------------#        
index_match    = np.asarray(list(itertools.product(list(np.arange(0,llen,1)),list(np.arange(0,dlen,1)))))    

LP = lrange[index_match[:,0]].copy()
DP = drange[index_match[:,1]].copy()
INDEX = []
U = []

for a in range(alen):
    for t in range(tlen):
        print(a,t)
            
        pi = arange[a] + DP-Q*LP-g(Q*LP,trange[t])
        
        lower = [i for i in range(len(pi)) if pi[i] > 0]    
        upper = [i for i in range(len(pi)) if pi[i] <= arange[a]-g(Q*LP[i],trange[t])-ebar*Q*LP[i] ]
        combined = list( set(lower) & set(upper) )
        INDEX.append(combined)
        U.append(pi[combined])
                

#-------------------------------#
#                               #
#   Value Function Iteration    #
#                               #    
#-------------------------------#

def Vnew(V):

    Vnew = np.zeros((alen,tlen))
    l_pol = np.zeros((alen,tlen))
    d_pol = np.zeros((alen,tlen))
    
    count = 0
    
    Vinterp = RegularGridInterpolator((arange,trange),V,bounds_error = False,fill_value = None)
    
    for a in range(alen):
        for t in range(tlen):
            
            lp = LP[INDEX[count]]
            dp = DP[INDEX[count]]
                
            #Group points by index
            s_high = np.vstack(( (1+erange[0])*lp-Rd*dp, (1-delta_theta)*trange[t] + phi(lp) ))
            s_low  = np.vstack(( (1+erange[1])*lp-Rd*dp, (1-delta_theta)*trange[t] + phi(lp) ))
                
            Vp = p[0]*Vinterp(s_high.T)+ p[1]*Vinterp(s_low.T)
                        
            V_today =  U[count]+beta*Vp
                                       
            l_pol[a,t] = lp[np.nanargmax(V_today)]
            d_pol[a,t] = dp[np.nanargmax(V_today)]
            Vnew[a,t]  = np.nanmax(V_today)    

            count = count + 1                
    #Howard Policy Improvement step        
    V0 = Vnew
    V1 = np.zeros((alen,tlen))
    h = 0
    while h < H:
        Vinterp = RegularGridInterpolator((arange,trange),V0,bounds_error = False,fill_value = None)
        
        for a in range(alen):
            for t in range(tlen):
        
                Vp = (p[0]*Vinterp(((1+erange[0])*l_pol[a,t]-Rd*d_pol[a,t],(1-delta_theta)*trange[t]+phi(l_pol[a,t])))+
                      p[1]*Vinterp(((1+erange[1])*l_pol[a,t]-Rd*d_pol[a,t],(1-delta_theta)*trange[t]+phi(l_pol[a,t]))) )

                V1[a,t] = u(arange[a]+d_pol[a,t]-Q*l_pol[a,t]-g(Q*l_pol[a,t],trange[t]))+beta*Vp
              
        V0 = V1.copy()
        h = h+1       
    
    Vnew = V0
            
    return Vnew , [l_pol,d_pol]

#Initialize VFI routine
error = 1
tol = 1e-08

Vguess = np.ones((alen,tlen))
V = Vguess 

looper = 0 
H = 0

#Run VFI routine to convergence
start = time.time()  
  
while error > tol:
    Vp = Vnew(V)[0]
    error = np.ma.masked_invalid(np.max(np.abs(Vp-V))) 
    print('Iteration:', looper)
    print('Error:',error)
    V = Vp.copy()    

    looper = looper + 1     
    
print(time.time() - start)   

#Extract policy functions
value = Vnew(V)[0]
policy = Vnew(V)[1]
Lp = policy[0]
Dp = policy[1]
 
#Compute profit and capital ratios
profit = np.zeros((alen,tlen))
capital = np.zeros((alen,tlen))
for a in range(alen):
    for t in range(tlen):
            
        profit[a,t] = arange[a]-Q*Lp[a,t]+Dp[a,t]-g(Q*Lp[a,t],trange[t])
        capital[a,t] = (Q*Lp[a,t]-Dp[a,t])/(Q*Lp[a,t]) 

#Compute state variable law of motion
ap_high = []
ap_low = []
for i in range(alen):
    ap_high.append( (1+erange[0])*Lp[i,0]-Rd*Dp[i,0]  )
    ap_low.append( (1+erange[1])*Lp[i,0]-Rd*Dp[i,0] )

#Plots
plt.close()
plt.figure(1)
plt.plot(arange,ap_high,label='high')
plt.plot(arange,ap_low,label='low')
plt.plot(arange,arange,linestyle='--')
plt.legend()

plt.figure(2)
plt.plot(arange,Lp[:,0],label='Lp')
plt.plot(arange,Dp[:,0],label='Dp')
plt.plot(arange,profit[:,0],label='profit')
plt.legend()

plt.figure(3)
[plt.plot(arange,capital[:,5*i]) for i in range(20)]
plt.axhline(ebar,linestyle='--')



