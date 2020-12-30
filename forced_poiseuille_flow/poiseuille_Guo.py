#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 09:58:34 2020

@author: bjerg
"""

saveRun = False

# import libraries
import numpy as np
import matplotlib.pyplot as plt
#from numba import njit, prange
import time
import pickle
from copy import deepcopy
from common import collide, stream, applyBC, updateMacro, compSource, getFeq


ReTarg = 50


Ny = 100
                 
Nx = Ny




max_steps = 50000               # maximum number of steps in time
nu        = 1/6                # kinematic viscosity
                      # inflow velocity (uniform x-component only)
Nout = 2000                     # log to screen every...
tol = 1e-18                     # steady-state tolerance
U = ReTarg*nu/Ny
Q   = 9                   # number of lattice directions
cs2 = 1/3                       # speed of sound ^2 
cs4 = cs2**2                    # speed of sound ^4

# position the actuator line

F = 2*U*nu/((Ny/2)**2)

Re = U*Ny/nu
print(Re)
print(U)
x0 = int(Nx/2)
# grid
Y, X = np.meshgrid(np.linspace(0.5,Ny-0.5,Ny),np.linspace(0.5,Nx-0.5,Nx))
Y = Y - Ny/2

tau         = nu/cs2 + 1/2 
omega       = 1/tau
omega_prime = 1 - omega

# compute and display Reynolds number based on actuator line radius


# allocate space for macroscopic variables and populations
rho  = np.ones((Nx,Ny))      # macroscopic density
ux   = np.zeros((Nx,Ny))    # velocity component x-dir.
uy   = np.zeros((Nx,Ny))     # velocity component y-dir.
feq  = np.zeros((Nx,Ny,Q))   # equilibrium distribution function
fin  = np.zeros((Nx,Ny,Q))   # distribution function
fout = np.zeros((Nx,Ny,Q))   # distribution function

meanKinOld = 1

# initial velocity: ux = U and uy=0 in whole domain
u2 = ux**2 + uy**2
fin[:,:,0] = 2/9*rho*(2-3*u2)
fin[:,:,1] = rho/18*(2 + 6*ux + 9*ux**2 - 3*u2)
fin[:,:,2] = rho/18*(2 + 6*uy + 9*uy**2 - 3*u2)
fin[:,:,3] = rho/18*(2 - 6*ux + 9*ux**2 - 3*u2)
fin[:,:,4] = rho/18*(2 - 6*uy + 9*uy**2 - 3*u2)
fin[:,:,5] = rho/36*(1 + 3*(ux + uy) + 9*ux*uy + 3*u2)
fin[:,:,6] = rho/36*(1 - 3*(ux - uy) - 9*ux*uy + 3*u2)
fin[:,:,7] = rho/36*(1 - 3*(ux + uy) + 9*ux*uy + 3*u2)
fin[:,:,8] = rho/36*(1 + 3*(ux - uy) - 9*ux*uy + 3*u2) 

# weigths and velocity sets
w  = np.array([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36]) 
cx = np.array([0,1,0,-1,0,1,-1,-1,1])
cy = np.array([0,0,1,0,-1,1,1,-1,-1])

# initialize external force source term
S = np.zeros((Nx,Ny,Q))
Fgauss = np.ones(X.shape)
#F = 0.2e-4
Fgauss = Fgauss*F


idx,jdx = np.unravel_index(np.arange(0,Nx*Ny),(Nx,Ny))

step_array = np.array([])
umax_array = np.array([])

# ////// main loop //////
start_time = time.time()
for step in range(1,max_steps):
    
    # source term for external forcing      
    #S = compSource(Nx,Ny,Q,tau,w,cx,ux,cs2,cs4,Fgauss)
    
    for i in range(0,9):
        S[:,:,i] = (1-1/(2*tau))*w[i]*( (cx[i]-ux)/cs2 + (cx[i]*ux[i]*cx[i])/cs4 )*F 
    
       
    # ///  collision  ///
    fout, u2 = collide(fin,feq,ux,uy,rho,omega,omega_prime,S)

    # /// streaming ///
    fin = stream(deepcopy(fout))  

    fin = applyBC(fin,fout,w,rho,cx,cy,ux,uy,cs2,cs4)
    # ///  boundary conditions  ///      
     
    rho, ux, uy = updateMacro(fin,Fgauss)
    ux += F/2
    # /// update macro. quant. ///
    
    # store
    step_array = np.append(step_array,step)
    umax_array = np.append(umax_array,ux[x0,int(Ny/2)])
    
    
    # log to screen
    if np.mod(step,Nout) == 0:  
        print(f'Step {step}, Elaps. time: {(time.time() - start_time):.2f}',flush=True) 
        plt.imshow(np.flipud(np.sqrt(u2).transpose()), cmap='jet', interpolation='nearest')
        plt.colorbar()
        plt.show()
        
        H = Ny
        plt.figure()
        plt.plot(ux[x0,:],Y[1,:],'b-')    
        plt.plot(-F/(2*nu)*(Y[x0,:]**2-(H/2)**2),Y[x0,:],'k-')
        plt.legend(['sim.','exact'])
    
        plt.show()
        
        plt.plot(step_array,umax_array,'r-')
        plt.xlabel('steps')
        plt.ylabel('u_max')
        plt.show()
        
        
yaxis = np.linspace(-0.5,0.5,200)
H = Ny
plt.figure(figsize=(10,4))
plt.plot(Y[1,:]/H,ux[x0,:]/U,'b-')    
plt.plot(Y[x0,:]/H,-F/(2*nu*rho[x0,:])*(Y[x0,:]**2-(H/2)**2)/U,'k-')
#plt.plot(yaxis,-F/(2*nu)*(yaxis**2-(1/2)**2)/U,'k-')
plt.legend(['simulation','exact'])
plt.ylabel('ux/Uc')
plt.xlabel('y/H')
plt.title('Horizontal velocity profile at x/H = 1')
plt.savefig('./figures/poiseuille_ux.png', bbox_inches='tight',dpi=300)    
    
print('/// DONE ///') # end the simulation

u = np.sqrt(ux**2+uy**2)
#fig = plt.figure(figsize=(10, 3))
#ax = fig.add_subplot(111)
#levels = np.linspace(np.min(u[~np.isnan(u)]),np.max(u[~np.isnan(u)]),30)
#plt.xlabel('x/H')
#plt.ylabel('y/H')
#plt.contourf(X/Nx,Y/Ny,u,levels,antialiased=True,cmap='jet')




fig = plt.figure(figsize=(10,5))
xticks = [0,0.25,0.5,0.75,np.max(u)/U]
im = plt.imshow(np.flipud(np.sqrt(u2).transpose()/np.max(u)),extent=[0,1,0,1], cmap='jet', interpolation='nearest')
im.set_clim([0, np.max(u)/U])
cbar = fig.colorbar(im,ticks=xticks, orientation="vertical", pad=0.05)

#cbar.set_label('|U|/Umax',rotation=270,labelpad=20,fontsize=14,y=0.50)
cbar.ax.set_yticklabels(['{:.2f}'.format(x) for x in xticks])


plt.xlabel('x/H')
plt.ylabel('y/H')

plt.savefig('./figures/poiseuille_Umag.png', bbox_inches='tight',dpi=300)




