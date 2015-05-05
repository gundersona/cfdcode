"""
Created on Mon Feb  2 20:13:44 2015
Anderson Sec7.3
CFD Solution for C-D Nozzle Using MacCormack's Technique
@author: gundersona
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

#Variables:
ntstep=1400 #number of time steps
k=1.4 #specific heat ratio
n=28 # total # of grid points
L=3 #total nozzle length
delx=L/(n-1) #delta x
x=np.linspace(0,L,n) #node locations along nozzle
#A=1+2.2*(x-1.5)**2 #area distribution of nozzle

A=np.zeros(n)
infile=open('cdnozzle_areas.txt','r')
count=0
for line in infile:
    A[count]=eval(line)
    count=count+1
infile.close()

rho=np.zeros((ntstep+1,n)) #dimensionless density
V=np.zeros((ntstep+1,n)) #dimensionless velocity
T=np.zeros((ntstep+1,n)) #dimensionless temperature
p=np.zeros((ntstep+1,n)) #dimensionless pressure
M=np.zeros((ntstep+1,n)) #mach number

delrho_p=np.zeros(n) #predictor partial derivatives
delV_p=np.zeros(n)
delT_p=np.zeros(n)

out=np.zeros((n,6)) #output matrix

#IC's:
for j in range(n):
    
    rho[0,j]=1-.3146*x[j]
    T[0,j]=1-.2314*x[j]
    V[0,j]=(.1+1.09*x[j])*T[0,j]**.5
    '''
    rho[0,j]=1-.014*x[j]
    T[0,j]=1-.008*x[j]
    V[0,j]=.03+.03*x[j]
    '''
#
#BEGIN MACCORMACK'S TECHNIQUE
#

for i in range(1,ntstep+1):
    
    #Specified BC's (2 on the inlet for subsonic flow)
    rho[i,0]=1
    T[i,0]=1
    
    #Calculate minimum time step
    delt=.5*delx/(T[i-1,0]**.5+V[i-1,0])
    for j in range(1,n):
        temp = .5*delx/(T[i-1,j]**.5+V[i-1,j])
        if temp < delt:
            delt = temp
    
    #Predictor Step
    for j in range(1,n-1):
        
        delrho_p[j]=-rho[i-1,j]*(V[i-1,j+1]-V[i-1,j])/delx- \
                    rho[i-1,j]*V[i-1,j]*(np.log(A[j+1])-np.log(A[j]))/delx- \
                    V[i-1,j]*(rho[i-1,j+1]-rho[i-1,j])/delx
                    
        delV_p[j]=-V[i-1,j]*(V[i-1,j+1]-V[i-1,j])/delx- \
                  1/k*((T[i-1,j+1]-T[i-1,j])/delx+
                        T[i-1,j]/rho[i-1,j]*(rho[i-1,j+1]-rho[i-1,j])/delx)
                        
        delT_p[j]=-V[i-1,j]*(T[i-1,j+1]-T[i-1,j])/delx- \
                  (k-1)*T[i-1,j]*((V[i-1,j+1]-V[i-1,j])/delx+ 
                                  V[i-1,j]*(np.log(A[j+1])-np.log(A[j]))/delx)
                                   
        rho[i,j]=rho[i-1,j]+delrho_p[j]*delt
        V[i,j]=V[i-1,j]+delV_p[j]*delt
        T[i,j]=T[i-1,j]+delT_p[j]*delt
  
    #Calculate predicted floating BC (only inflow necessary for corrector)
    V[i,0]=2*V[i,1]-V[i,2]
    
    #Corrector Step
    for j in range(1,n-1):
        
        delrho_c=-rho[i,j]*(V[i,j]-V[i,j-1])/delx- \
                    rho[i,j]*V[i,j]*(np.log(A[j])-np.log(A[j-1]))/delx- \
                    V[i,j]*(rho[i,j]-rho[i,j-1])/delx
                    
        delV_c=-V[i,j]*(V[i,j]-V[i,j-1])/delx- \
                  1/k*((T[i,j]-T[i,j-1])/delx+
                        T[i,j]/rho[i,j]*(rho[i,j]-rho[i,j-1])/delx)
                        
        delT_c=-V[i,j]*(T[i,j]-T[i,j-1])/delx- \
                  (k-1)*T[i,j]*((V[i,j]-V[i,j-1])/delx+ 
                                   V[i,j]*(np.log(A[j])-np.log(A[j-1]))/delx)
                                   
        delrho_ave=.5*(delrho_p[j]+delrho_c)
        delV_ave=.5*(delV_p[j]+delV_c)
        delT_ave=.5*(delT_p[j]+delT_c)
                                   
        rho[i,j]=rho[i-1,j]+delrho_ave*delt
        V[i,j]=V[i-1,j]+delV_ave*delt
        T[i,j]=T[i-1,j]+delT_ave*delt
        
    #Calculate corrected floating BC's
    #Inflow boundary:
    V[i,0]=2*V[i,1]-V[i,2]
    #Outflow boundary:
    V[i,n-1]=2*V[i,n-2]-V[i,n-3]    
    rho[i,n-1]=2*rho[i,n-2]-rho[i,n-3]
    T[i,n-1]=2*T[i,n-2]-T[i,n-3]
    
    #Calculate pressure and mach number
    for j in range(n):
        p[i,j]=rho[i,j]*T[i,j]
        M[i,j]=V[i,j]/T[i,j]**.5
        
#Create output table
for j in range(n):
    out[j,0]=A[j]
    out[j,1]=rho[i,j]
    out[j,2]=V[i,j]
    out[j,3]=T[i,j]
    out[j,4]=p[i,j]
    out[j,5]=M[i,j]
np.savetxt('cdnozzle_out.csv',out,delimiter=',')
                          
                                   
    
    