# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 10:37:50 2017

@author: jordan
"""

import numpy as np
from scipy.optimize import fsolve
import numdifftools as nd

class MultiBank_Model:
    
    def __init__(self,I,J,beta,chi,tau,kappa,s,alpha,Abar,delta,gamma,rho):
        self.I       = I
        self.J       = J        
        self.beta    = beta 
        self.chi     = chi
        self.tau     = tau
        self.kappa   = kappa
        self.s       = s
        self.alpha   = alpha
        self.Abar    = Abar
        self.delta   = delta
        self.gamma   = gamma
        self.P       = rho*np.eye(J)
    
    #Read in some common functions
    def phi(self,D,Dbar):
        return D+self.kappa*(D-Dbar)**2
    
    def phid(self,D,Dbar):
        return 1+2*self.kappa*(D-Dbar)
        
    def g(self,L,s):
        return (L*s)**2
        
    def gd(self,L,s):
        return 2*L*s**2

    def f(self,A,K,alpha):
        return A*K**alpha
        
    def fd(self,A,K,alpha):
        return A*alpha*K**(alpha-1)    
        
    def uc(self,c):
        if self.gamma == 1:
            return 1/c
        elif self.gamma != 1:
            return c**(-self.gamma)
    
    def SS_System(self,x):
        """
        Given the number of intermediaries and number of firms, this function
            returns the steady state values of the equilibrium objects.        
        
        x[0:I*J]          : Bank loans (L_1^1, L_1^2,...,L_1^J, L_2^1,...,L_I^j)
        x[I*J:I*J+I]      : Bank debt with household (d_1',...,d_I')
        x[I*J+I:I*J+2*I]  : Bank dividends (D_1,...,D_I)
        x[I*J+2*I:I*J+3*I]: Bank steady state dividends (Dbar1,...,DbarI)
        x[I*J+3*I:I*J+4*I]: Bank equity constraint multiplier (mu1,...,muI)
    
        M=I*J+4*I
    
        x[M:M+J]          : Firm Loan interest rate (R^1,...,R^J)
        x[M+J:M+2*J]      : Firm Profits (pi^1,...,pi^J)
        x[M+2*J:M+3*J]    : Firm capital (K1,...,KJ)
        x[M+3*J:M+4*J]    : Firm loans (L1,...LJ)
    
        N= M+4*J
    
        x[N]              : c
        x[N+1]            : R
        x[N+2]            : d_house
        """
        I = self.I
        J = self.J
        
        if I*J != len(self.s):
            print('Length of s vector not equal to IJ')
        if J != len(self.Abar):
            print('Length of Abar not equal to J')
        if J != len(self.alpha):
            print('Length of alpha vector not equal to J')
        
        M = I*J+4*I
        N = M+4*J
                            #Intermediary Conditions
        #Debt Optimality
        cond1 = self.beta*x[N+1]+self.chi*x[I*J+3*I:I*J+4*I]*self.phid(x[I*J+I:I*J+2*I],x[I*J+2*I:I*J+3*I])-1
        
        #Loan Optimality
        cond2 = (self.beta*np.tile( (x[M:M+J]-self.tau)/(1-self.tau)  ,I)+
                    np.repeat(x[I*J+3*I:I*J+4*I]*self.phid(x[I*J+I:I*J+2*I],x[I*J+2*I:I*J+3*I]),J)-
                        1-self.gd(x[0:I*J],self.s)
                        )       
        #Budget Constraint
        gross_return = np.tile( (x[M:M+J]-self.tau)/(1-self.tau)  ,I)*x[0:I*J]
        loan_purchase = x[0:I*J] + self.g(x[0:I*J],self.s)
        loan_EC       = x[0:I*J]
        
        returns = np.zeros((I))
        loans   = np.zeros((I))
        loans_EC = np.zeros((I))
        
        for i in range(I):
            returns[i]    = np.sum(gross_return[i*J:i*J+J])
            loans[i]      = np.sum(loan_purchase[i*J:i*J+J])
            loans_EC[i]   = np.sum(loan_EC[i*J:i*J+J])
    
        cond3 = (x[I*J:I*J+I] + returns - self.phi(x[I*J+I:I*J+2*I],x[I*J+2*I:I*J+3*I])-
                    np.repeat(x[N+1],I)*x[I*J:I*J+I]-loans
                            )
        #Equity Constraint
        cond4 = loans_EC - self.chi*x[I*J:I*J+I] 
    
                                #Household Conditions
        #Euler Equation
        cond5 = self.beta-(1-self.tau)/(x[N+1]-self.tau)

        #Budget Constraint
    
        profits = np.sum(x[M+J:M+2*J])
    
        T = (((x[N+1]-self.tau)/(1-self.tau)-x[N+1] )*np.sum(x[I*J:I*J+I])+     #HH Debt Refund
                    np.sum(np.tile((x[M:M+J]-self.tau)/(1-self.tau)-x[M:M+J],I)*x[0:I*J]) #Loan Debt Refund
                    )
                
        cond6 = x[N] + x[N+2]-((x[N+1]-self.tau)/(1-self.tau))*x[N+2]-np.sum(x[I*J+I:I*J+2*I])+T-profits      

                            #Firm Conditions
        #Loan Optimality
        cond7 = self.beta*x[M:M+J] -  1
    
        #Capital Optimality
        cond8 = self.beta*(self.fd(self.Abar,x[M+2*J:M+3*J],self.alpha)+1-self.delta)-1
    
        #Budget Constraint
        cond9 = (self.f(self.Abar,x[M+2*J:M+3*J],self.alpha)+x[M+3*J:M+4*J]+
                    (1-self.delta)*x[M+2*J:M+3*J]-x[M:M+J]*x[M+3*J:M+4*J]-
                        x[M+J:M+2*J]-x[M+2*J:M+3*J] 
                                )
                            #Market Clearing
        #Dividends    
        cond10 = x[I*J+I:I*J+2*I]-x[I*J+2*I:I*J+3*I]
        
        #Loans
        loan_EC1     = x[0:I*J]    
        loans_clear = np.zeros((J))
        for j in range(J):
            loans_clear[j] = np.sum(loan_EC1[j::J])
            
        cond11 = x[M+3*J:M+4*J] - loans_clear
            
        #Household Debt
        cond12 = x[N+2] - np.sum(x[I*J:I*J+I])    
    
        return np.hstack((cond1,cond2,cond3,cond4,cond5,cond6,cond7,cond8,cond9,cond10,cond11,cond12  ))
    
    def state_count(self):
        return 2+self.I+3*self.J+self.I*self.J
        
    def endog_count(self):
        return 1+2*self.I+self.J
                
    def exog_count(self):
        return self.J        
            
    def ss_var_count(self):
        """
        Returns the number of variables used to solve for steady state
        """
        return 3+4*(self.I+self.J)+self.I*self.J
            
    def eq_var_count(self):
        """
        Returns the number of variabs used to solve for equilibrium system
        """
        return 2*(3+5*self.J+3*self.I+self.I*self.J)


    def SS_Solver(self,inits):
        """
        Returns steady state values, given initial conditions
        """
        return fsolve(self.SS_System,inits)
            
    def SS_Output(self,inits):
        """
        Print steady state output values
        """
        I = self.I
        J = self.J            
        M = I*J+4*I
        N = M+4*J
        root = self.SS_Solver(inits)
            
        print('Bank Loans', root[0:I*J])
        print('Bank Debt',  root[I*J:I*J+I])  
        print('Bank Dividends',  root[I*J+I:I*J+2*I])
        print('EC Multipliers', root[I*J+3*I:I*J+4*I])
        print('') 
        print('Firm Profits', root[M+J:M+2*J])    
        print('Firm Capital', root[M+2*J:M+3*J]) 
        print('Firm Agg Loans', root[M+3*J:M+4*J])
        print('')
        print('Household Debt',root[N+2])   
        print('Consumption',root[N])
        print('')
        print('Firm Interest Rates', root[M:M+J])   
        print('Debt Interest',root[N+1])
        
    def SS_Dividends(self,inits):
        """
        Returns steady state dividends, given initial conditions 
        """            
        I = self.I
        J = self.J
        root = self.SS_Solver(inits)
        
        return np.array((root[I*J+I:I*J+2*I] ))

    def Equilibrium_System(self,x,Dbar):
        """
        STATE VARS, TODAY
        x[0:I*J]              : Bank loans (L_1^1, L_1^2,...,L_1^J, L_2^1,...,L_I^j)
        x[I*J:I*J+I]          : Bank debt with household (d_1,...,d_I)
        x[I*J+I:I*J+I+J]      : Firm Loan interest rate (R^1,...,R^J)
        x[I*J+I+J:I*J+I+2*J]  : Firm loans (L1,...LJ) 
        x[I*J+I+2*J:I*J+I+3*J]: Firm capital (K1,...,KJ)
        x[I*J+I+3*J]          : R
        x[I*J+I+3*J+1]        : d_house      
    
        M = I*J+I+3*J+2

        ENDOGENOUS VARS, TODAY    
        x[M:M+I]                  : Bank dividends (D_1,...,D_I)
        x[M+I:M+2*I]              : Bank equity constraint multiplier (mu1,...,muI)
        x[M+2*I:M+2*I+J]          : Firm Profits (pi^1,...,pi^J)
        x[M+2*I+J]                : c
    
        N = M+2*I+J+1
    
        STATE VARS, TOMORROW
        x[N:N+I*J]                : Bank loans (L_1^1', L_1^2',...,L_1^J', L_2^1',...,L_I^J')
        x[N+I*J:N+I*J+I]          : Bank debt with household (d_1',...,d_I')
        x[N+I*J+I:N+I*J+I+J]      : Firm Loan interest rate (R^1',...,R^J')
        x[N+I*J+I+J:N+I*J+I+2*J]  : Firm loans (L1',...LJ') 
        x[N+I*J+I+2*J:N+I*J+I+3*J]: Firm capital (K1',...,KJ')
        x[N+I*J+I+3*J]            : R'   
        x[N+I*J+I+3*J+1]          : d_house'

        MM = N+I*J+I+3*J+2
    
        ENDOGENOUS VARS, TOMORROW
        x[MM:MM+I]                : Bank dividends (D_1',...,D_I')
        x[MM+I:MM+2*I]            : Bank equity constraint multiplier (mu1',...,muI')
        x[MM+2*I:MM+2*I+J]        : Firm Profits (pi^1',...,pi^J')
        x[MM+2*I+J]               : c'
    
        NN = MM+2*I+J+1
    
        STOCHASTIC VARS, TODAY
        x[NN:NN+J]                : TFP factors (A^1,...,A^J)
    
        STOCHASTIC VARS, TOMORROW
        x[NN+J:NN+2*J]            : TFP Factors(A^1',...,A^J')
        """    
        I = self.I
        J = self.J
        
        M = I*J+I+3*J+2
        N = M+2*I+J+1
        MM = N+I*J+I+3*J+2
        NN = MM+2*I+J+1
    
        #Bank SDF
        m = self.beta*(self.uc(x[MM+2*I+J])/self.uc(x[M+2*I+J]) )*(self.phid(x[M:M+I],Dbar)/self.phid(x[MM:MM+I] ,Dbar))    
    
                            #Intermediary Conditions
        #Debt Optimality
        cond1 = m*x[N+I*J+I+3*J]+self.chi*x[M+I:M+2*I]*self.phid(x[M:M+I],Dbar)-1
    
        #Loan Optimality
        cond2 = np.repeat(m,J)*np.tile((x[N+I*J+I:N+I*J+I+J]-self.tau)/(1-self.tau),I)+np.repeat(x[M+I:M+2*I]*self.phid(x[M:M+I],Dbar),J)-1-self.gd(x[N:N+I*J],self.s)
    
        #Budget Constraint
        gross_return = x[0:I*J]*np.tile( ((x[I*J+I:I*J+I+J])-self.tau)/(1-self.tau),I )
        loan_purchase = x[N:N+I*J]+self.g(x[N:N+I*J],self.s)
        loan_EC       = x[N:N+I*J] 
    
        returns = np.zeros((I))
        loans   = np.zeros((I))
        loans_EC = np.zeros((I))


        for i in range(I):
            returns[i]    = np.sum(gross_return[i*J:i*J+J])
            loans[i]      = np.sum(loan_purchase[i*J:i*J+J])
            loans_EC[i]   = np.sum(loan_EC[i*J:i*J+J])
    
        cond3 = x[N+I*J:N+I*J+I] + returns- self.phi(x[M:M+I],Dbar)- x[I*J+I+3*J]*x[I*J:I*J+I]-loans    
    
        #Equity Constraint
        cond4 = loans_EC - self.chi*x[N+I*J:N+I*J+I]  
    
                                #Household Conditions
        #Euler Equation
        cond5 = self.beta*(self.uc(x[MM+2*I+J])/self.uc(x[M+2*I+J]) ) - (1-self.tau)/(x[N+I*J+I+3*J]-self.tau)
    
        #Budget Constraint
        T =( ( (x[I*J+I+3*J] -self.tau)/(1-self.tau)-x[I*J+I+3*J] )*np.sum(x[I*J:I*J+I])+
                    np.sum(np.tile( (x[I*J+I:I*J+I+J]-self.tau)/(1-self.tau) - x[I*J+I:I*J+I+J],I )*x[0:I*J])
                        )
                    
        profits = np.sum(x[M+2*I:M+2*I+J])
    
        cond6 = x[M+2*I+J] + x[N+I*J+I+3*J+1] - ((x[I*J+I+3*J]-self.tau)/(1-self.tau))*x[I*J+I+3*J+1]-np.sum(x[M:M+I])+T-profits
    
                                #Firm Conditions
        #Loan Optimality
        cond7 = self.beta*(self.uc(x[MM+2*I+J])/self.uc(x[M+2*I+J]) )*x[N+I*J+I:N+I*J+I+J]-1
    
        #Capital Optimality
        cond8 = self.beta*(self.uc(x[MM+2*I+J])/self.uc(x[M+2*I+J]) )*(self.fd(x[NN+J:NN+2*J],x[N+I*J+I+2*J:N+I*J+I+3*J],self.alpha)+1-self.delta)-1
    
        #Budget Constraint
        cond9 = ( self.f(x[NN:NN+J],x[I*J+I+2*J:I*J+I+3*J],self.alpha)+                       #output
                        x[N+I*J+I+J:N+I*J+I+2*J]+(1-self.delta)*x[I*J+I+2*J:I*J+I+3*J]-   #new loans + depreciated capital
                            x[I*J+I:I*J+I+J]*x[I*J+I+J:I*J+I+2*J]-                  #Repay old loans
                                x[M+2*I:M+2*I+J]-                                   #Profits
                                    x[N+I*J+I+2*J:N+I*J+I+3*J]  )                   #New capital
    
                                #Market Clearance
        #Loans
        loan_EC1     = x[N:N+I*J]     
        loans_clear = np.zeros((J))
        for j in range(J):
            loans_clear[j] = np.sum(loan_EC1[j::J])
        
        cond10 = x[N+I*J+I+J:N+I*J+I+2*J] - loans_clear
    
        #Household debt
        cond11 = x[N+I*J+I+3*J+1] - np.sum(x[N+I*J:N+I*J+I] )
    
        return np.hstack((cond1,cond2,cond3,cond4,cond5,cond6,cond7,cond8,cond9,cond10,cond11))


    def Jacobian(self,ss_values,ss_dividends):
            
        eq_sys = lambda x: self.Equilibrium_System(x,ss_dividends)

        return nd.Jacobian(eq_sys)(ss_values)
        
    def AC_compute(self,Jac):
        """
        Jac: Jacobian matrix, evaluated at steady state
        
        Returns linearized coefficient matrices
        """
        
        nx = 2+self.I+3*self.J+self.I*self.J
        nc = 1+2*self.I+self.J
        nz = self.J        
        
        A1 = Jac[:,(nc+nx):2*(nc+nx)]   #Shape: (eqm conds) x (nx+nc = n) where eqm conds = n 
        A2 = Jac[:,0:(nc+nx)]    #same
        B2 = Jac[:,2*(nc+nx)+nz:2*(nc+nx)+2*nz]   #shape: (eqm conds) x (nz)
        B1 = Jac[:,2*(nc+nx):2*(nc+nx)+nz]   #same
    
        #Generalized Eigenvalue problem
        from scipy.linalg import eig
        import scipy.linalg as la
        eig_val, eig_vec = eig(A2,-A1)[0], eig(A2,-A1)[1]

        inside = np.argsort(np.abs(eig_val))[0:nx]
        outside =  np.argsort(np.abs(eig_val))[nx:nx+nc]

        if len(inside) != nx:
            stop= """
        
        
                Number of eigenvalues inside unit root does not match number of states!
                
                
                """
            print(stop)
    
        eig_index = np.concatenate(( inside , outside ))

        #Rearranged
        eig_order = eig_val[eig_index]
        eig_vec_order = eig_vec[:,eig_index]

        V11 = eig_vec_order[0:nx,0:nx]
        #V12 = eig_vec_order[0:nx,nx:(nx+nc)]
        V21 = eig_vec_order[nx:(nx+nc),0:nx]
        #V22 = eig_vec_order[nx:(nx+nc),nx:(nx+nx)]
        Del1 = np.eye(nx)*eig_order[0:nx]
        
        A = np.dot( np.dot(V11,Del1) , la.inv(V11))
        C = np.dot(V21,la.inv(V11))
    
        return [A,C,A1,A2,B1,B2]


    def BD_System(self,x,A1,A2,B1,B2,C,P):
        
        P = self.P
        
        nx = np.shape(C)[1]
        nc = np.shape(C)[0]
        nz = np.shape(B1)[1]        
    
        Bx = x[:nx*nz]   #coefficients for B
        Dx = x[-nc*nz:]  #coefficients for D
    
        Bx = np.reshape(Bx,(nx,nz))
        Dx = np.reshape(Dx,(nc,nz))
    
        #first = np.tile(Bx,(2,1))

        Bxp = np.dot(C,Bx) + np.dot(Dx,P)   
        
        first = np.vstack( (Bx,Bxp) )
        
        zeroed = np.zeros((nx,nz))
        
        second = np.vstack( (zeroed,Dx) )
        
        full = np.dot(A1,first)+np.dot(A2,second)+B1+np.dot(B2,P)
        
        vectorized = np.reshape(full, (nx+nc)*nz  )
        
        return list(vectorized )


    def Policy_Function(self,Jac,state_ss,endog_ss,exog_ss):
        """
        Returns the policy functions for both the state and endogenous variables
        
        Inputs: Jacobian around steady state and the steady state values,
            partitioned by state, engogenous and exogenous variable categories.
        """
        
        AC_solution = self.AC_compute(Jac)

        A  = AC_solution[0]
        C  = AC_solution[1]
        A1 = AC_solution[2]
        A2 = AC_solution[3]
        B1 = AC_solution[4]
        B2 = AC_solution[5]

        BD_System_final = lambda x: self.BD_System(x,A1,A2,B1,B2,C,self.P)
        
        nx = 2+self.I+3*self.J+self.I*self.J
        nc = 1+2*self.I+self.J
        nz = self.J  
        
        inits = list(np.zeros(nx*nz+nc*nz))
        
        solution = fsolve(BD_System_final,inits)

        B = solution[:nx*nz]
        B = np.reshape(B,(nx,nz))
        D = solution[-nc*nz:]
        D = np.reshape(D,(nc,nz))
        
        G = lambda x,shock:  state_ss + np.dot(A,x-state_ss)+np.dot(B,shock-exog_ss) #state equilibrium law of motion
        H = lambda x,shock:  endog_ss + np.dot(C,x-state_ss)+np.dot(D,shock-exog_ss) #control function    
        
        return [G,H]