# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 13:22:02 2017

@author: jordan
"""

import numpy as np
import matplotlib.pyplot as plt
import Equilibrium_Class as ec

#----------------------------#
#                            #   
#   Initialize parameters    #
#                            #
#----------------------------#
#Standard parameters
beta  = .97
tau   = .35
kappa = 0
chi   = 2
gamma = 1
rho   = .9
delta = .1

#Model-specific parameterization
I = 3
J = 3

s1 = np.array((.01,.1,.1))
s2 = np.array((.1,.01,.1))
s3 = np.array((.1,.1,.01))
s  = np.hstack((s1,s2,s3))

alpha = np.array((.3,.3,.3))
Abar  = np.array((2,2,2))

#-----------------------------------------#
#                                         #
#   Solving for Model Policy Functions    #
#                                         #
#-----------------------------------------#
#Initiative object
Model = ec.MultiBank_Model(I,J,beta,chi,tau,kappa,s,alpha,Abar,delta,gamma,rho)

ss_count = Model.ss_var_count()    #Number of variables for calculating steady state

#Stack steady state values for full equilibrium system
SS_values    = Model.SS_Solver(.4*np.ones(ss_count))
M = I*J+4*I
N = M+4*J
state_ss = np.hstack((SS_values[0:I*J],SS_values[I*J:I*J+I],SS_values[M:M+J],SS_values[M+3*J:M+4*J],SS_values[M+2*J:M+3*J],SS_values[N+1],SS_values[N+2] ))
endog_ss = np.hstack(( SS_values[I*J+I:I*J+2*I],SS_values[I*J+3*I:I*J+4*I],SS_values[M+J:M+2*J],SS_values[N] ))
exog_ss  = Abar
total_ss = np.hstack((state_ss,endog_ss,state_ss,endog_ss,exog_ss,exog_ss))

SS_dividends = Model.SS_Dividends(.4*np.ones(ss_count))    #steady state dividends

SS_jacobian = Model.Jacobian(total_ss,SS_dividends)   #Linearize equilibrium system around steady state

#Solve for model policy functions
policy = Model.Policy_Function(SS_jacobian,state_ss,endog_ss,exog_ss) 
state_policy = policy[0]    #state variables functions
endog_policy = policy[1]    #endogenous varibales functions

#--------------------------------------------------------#
#                                                        #   
#   Comparative Statics and Impulse Resonse Functions    #
#                                                        #
#--------------------------------------------------------#
"""
If x contains the stacked model variables, partitioned by state variables and
    endogneous variables, respectively, then their indices correspond to

        x[0:I*J]              : Bank loans (L_1^1, L_1^2,...,L_1^J, L_2^1,...,L_I^j)
        x[I*J:I*J+I]          : Bank debt with household (d_1,...,d_I)
        x[I*J+I:I*J+I+J]      : Firm Loan interest rate (R^1,...,R^J)
        x[I*J+I+J:I*J+I+2*J]  : Firm loans (L1,...LJ) 
        x[I*J+I+2*J:I*J+I+3*J]: Firm capital (K1,...,KJ)
        x[I*J+I+3*J]          : R
        x[I*J+I+3*J+1]        : d_house      
    
        M = I*J+I+3*J+2
    
        x[M:M+I]                  : Bank dividends (D_1,...,D_I)
        x[M+I:M+2*I]              : Bank equity constraint multiplier (mu1,...,muI)
        x[M+2*I:M+2*I+J]          : Firm Profits (pi^1,...,pi^J)
        x[M+2*I+J]                : c
"""

N = 100 #Number of simulation periods
shock = np.array((1,0,0))  #Percentage shock deviation for IRF

if len(shock) != J:
    print('Shock vector doesn\'t match number of TFP factors!')

def shock_generate(shock,N):
    """
    shock  : J-vector containing the % size of shock to TFP factors 
    N      : Number of periods in simulation
    Returns: NxJ-dimensional shock sequence 
    """

    #Shock in period int(n/10)
    shock_size =  (shock/100)*Abar
    
    #Compute AR pieces for rest of period
    AR_lags = shock_size*np.concatenate((
            np.zeros((10,len(shock))),
            np.tile(rho**np.arange(0,N-int(N/10)),(len(shock),1)).T
            ))
    
    return np.tile(Abar,(N,1) ) + AR_lags
    

state          = np.zeros((N,Model.state_count() ))
control        = np.zeros((N,Model.endog_count() ))
shock_sequence = shock_generate(shock,N)
state[0]       = state_ss

for i in range(N):
    if i != N-1:
        
        state[i+1,] = state_policy(state[i,],shock_sequence[i,])
        control[i,] = endog_policy(state[i,],shock_sequence[i,])

#Percentage Deviation from Steady State
state_dev = 100*((state-state_ss)/state_ss)
endog_dev = 100*((control-endog_ss)/endog_ss)
shock_dev = 100*((shock-exog_ss)/exog_ss)



plt.figure(1)
plt.subplot(2,2,1) 
plt.plot(state_dev[1:N,0:I*J])
plt.ylabel('% Deviation from Steady State',fontsize=20)
plt.title('Firm Loans')
plt.subplot(2,2,2)
plt.plot(state_dev[1:N,I*J:I*J+I])
plt.ylabel('% Deviation from Steady State',fontsize=20)
plt.title('Bank Household Debt')
plt.subplot(2,2,3)
plt.plot(state_dev[1:N,I*J+I+J:I*J+I+2*J])
plt.ylabel('% Deviation from Steady State',fontsize=20)
plt.title('Firm Aggregate Loans')
plt.subplot(2,2,4)
plt.plot(state_dev[1:N,I*J+I+2*J:I*J+I+3*J])
plt.ylabel('% Deviation from Steady State',fontsize=20)
plt.title('Firm Aggregate Capital')
plt.suptitle('%s Percentage Shock to TFP Factors'%shock,fontsize=20)







