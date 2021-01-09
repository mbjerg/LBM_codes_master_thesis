#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lattice Boltzmann Method CFD
Flow case: Backward-facing step
Author: Morten bjerg

"""

# import libraries
import numpy as np
import matplotlib.pyplot as plt
from numba import njit 
import time
from copy import deepcopy
import pickle

# set True to save result in a pickle
saveRun = False

ReTarg = 389          # target Reynolds number
NH     = 30           # number of grid points across channel 
AR     = 20
S      = int(0.94*NH) # step length
Ny     = NH + S
Nx     = AR*NH     

# control parameters
max_steps = int(100e3) # maximum number of steps in time
nu        = 0.01       # kinematic viscosity
U  = ReTarg*nu/(2*NH)  # inlet mean velocity
Q         = 9          # number of lattice directions
cs2 = 1/3              # speed of sound ^2 
cs4 = cs2**2           # speed of sound ^4
Nout = 500             # log to screen every...
tol = 1e-9             # steady-state tolerance

# inlet profile
inaxis = np.arange(-NH/2,NH/2) + 0.5
inlet_prof = 3/2*U*(1-(2*inaxis/NH)**2) 

# grid
Y, X = np.meshgrid(np.linspace(0.5,Ny-0.5,Ny),np.linspace(0.5,Nx-0.5,Nx))

# relaxation rate
tau         = nu/cs2 + 1/2 
omega       = 1/tau
omega_prime = 1 - omega

# compute and display Reynolds number
Re = np.abs(2*NH*U/nu)
print(f'Reynolds number: {Re:.0f}')

# allocate space for macroscopic variables and populations
rho  = np.ones((Nx,Ny))      # macroscopic density
ux   = np.zeros((Nx,Ny))     # velocity component x-dir.
uy   = np.zeros((Nx,Ny))     # velocity component y-dir.
feq  = np.zeros((Nx,Ny,Q))   # equilibrium distribution function
fin  = np.zeros((Nx,Ny,Q))   # distribution function
fout = np.zeros((Nx,Ny,Q))   # distribution function

# get linear indices such that Njit can loop over matrix with one for-loop
idx,jdx = np.unravel_index(np.arange(0,Nx*Ny),(Nx,Ny))

# weigths and velocity sets
w  = np.array([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36]) 
cx = np.array([0,1,0,-1,0,1,-1,-1,1])
cy = np.array([0,0,1,0,-1,1,1,-1,-1])

# initialization
for i in range(Q):
    fin[:,:,i] = w[i]
    
# exclude small corner below inlet channel
ux[:S,:S] = np.nan
uy[:S,:S] = np.nan 
fin[:S,:S,:] = np.nan

    
@njit()
def collide(f,feq,ux,uy,rho,omega,omega_prime):
    u2 = ux**2 + uy**2
    feq[:,:,0] = 2/9*rho*(2-3*u2)
    feq[:,:,1] = rho/18*(2 + 6*ux + 9*ux**2 - 3*u2)
    feq[:,:,2] = rho/18*(2 + 6*uy + 9*uy**2 - 3*u2)
    feq[:,:,3] = rho/18*(2 - 6*ux + 9*ux**2 - 3*u2)
    feq[:,:,4] = rho/18*(2 - 6*uy + 9*uy**2 - 3*u2)
    feq[:,:,5] = rho/36*(1 + 3*(ux + uy) + 9*ux*uy + 3*u2)
    feq[:,:,6] = rho/36*(1 - 3*(ux - uy) - 9*ux*uy + 3*u2)
    feq[:,:,7] = rho/36*(1 - 3*(ux + uy) + 9*ux*uy + 3*u2)
    feq[:,:,8] = rho/36*(1 + 3*(ux - uy) - 9*ux*uy + 3*u2) 
    
    return f*omega_prime + feq*omega, u2

def stream(f):
    f[:,:,1] = np.roll(f[:,:,1],[+1,+0],axis=(0,1))  
    f[:,:,2] = np.roll(f[:,:,2],[+0,+1],axis=(0,1))
    f[:,:,3] = np.roll(f[:,:,3],[-1,+0],axis=(0,1))
    f[:,:,4] = np.roll(f[:,:,4],[+0,-1],axis=(0,1))
    f[:,:,5] = np.roll(f[:,:,5],[+1,+1],axis=(0,1))
    f[:,:,6] = np.roll(f[:,:,6],[-1,+1],axis=(0,1))
    f[:,:,7] = np.roll(f[:,:,7],[-1,-1],axis=(0,1))
    f[:,:,8] = np.roll(f[:,:,8],[+1,-1],axis=(0,1))
    
    return f


# start
start_time = time.time()
meanKinOld = 1
uxprofOld = np.ones(Ny)

# -- main loop --
for step in range(1,max_steps):  
    
    # ///  collision  ///
    fout, u2 = collide(fin,feq,ux,uy,rho,omega,omega_prime)
      
    # /// streaming ///
    fin = stream(deepcopy(fout))  
    
    # ///  boundary conditions  ///   
    # BB inlet
    fin[0,S:,5] = fout[0,S:,7] - 2*w[7]*rho[0,S:]*cx[7]*inlet_prof/cs2
    fin[0,S:,1] = fout[0,S:,3] - 2*w[3]*rho[0,S:]*cx[3]*inlet_prof/cs2
    fin[0,S:,8] = fout[0,S:,6] - 2*w[6]*rho[0,S:]*cx[6]*inlet_prof/cs2
        
    # anti-BB outlet
    uxw = ux[-1,:] + 1/2*(ux[-1,:] - ux[-2,:])
    uyw = uy[-1,:] + 1/2*(uy[-1,:] - uy[-2,:])
    u2w =  uxw**2 + uyw**2
    
    fin[-1,:,7] = -fout[-1,:,5] + 2*w[5]*rho[-1,:]*(1 + (cx[5]*uxw + cy[5]*uyw)**2/(2*cs4) - u2w/(2*cs2))
    fin[-1,:,3] = -fout[-1,:,1] + 2*w[1]*rho[-1,:]*(1 + (cx[1]*uxw + cy[1]*uyw)**2/(2*cs4) - u2w/(2*cs2))
    fin[-1,:,6] = -fout[-1,:,8] + 2*w[8]*rho[-1,:]*(1 + (cx[8]*uxw + cy[8]*uyw)**2/(2*cs4) - u2w/(2*cs2))
   
    # step-wall
    fin[:S,S,6] = fout[:S,S,8]
    fin[:S,S,2] = fout[:S,S,4]
    fin[:S,S,5] = fout[:S,S,7] 
    
    fin[S,:S,5] = fout[S,:S,7] 
    fin[S,:S,1] = fout[S,:S,3] 
    fin[S,:S,8] = fout[S,:S,6] 
    
    # step-wall corner
    fin[S,S,5] = fin[S,S,7] 
       
    # BB top wall
    fin[:,-1,8] = fout[:,-1,6] 
    fin[:,-1,4] = fout[:,-1,2] 
    fin[:,-1,7] = fout[:,-1,5] 
   
    # BB bottom wall
    fin[:,0,6] = fout[:,0,8]
    fin[:,0,2] = fout[:,0,4]
    fin[:,0,5] = fout[:,0,7]   
        
    fin[:S,:S,:] = np.nan
          
    # /// update macro. quant. ///
    rho = np.sum(fin,2)
    ux  = (np.sum(fin[:,:,[1,5,8]],2) - np.sum(fin[:,:,[3,6,7]],2))/rho
    uy  = (np.sum(fin[:,:,[2,5,6]],2) - np.sum(fin[:,:,[4,7,8]],2))/rho
            
    # log to screen
    if np.mod(step,Nout) == 0:
        
        meanKin = np.mean(u2[~np.isnan(u2)]/2) # compute mean kinetic energy (rho = 1)
        err = np.abs(meanKin-meanKinOld)/Nout/meanKinOld # compute relative error
        print(f'Step {step}, Elaps. time: {(time.time() - start_time):.2f}, Err: {err:.6f}')
        if  err < tol:
            break
        else:
            meanKinOld = meanKin
            plt.imshow(np.flipud(np.sqrt(u2).transpose()), cmap='jet', interpolation='nearest')
            plt.show()
           
print('/// DONE ///') # end the simulation

# save simulation result as pickle in ./data
if saveRun:
    filename = f'./data/Re{Re:.0f}_{Nx:.0f}x{Ny:.0f}.pickle'    
    with open(filename,'wb') as f:
        pickle.dump([X,Y,U,rho,ux,uy,Nx,Ny,NH,S,step],f)
        
    
plt.imshow(np.flipud(np.sqrt(u2).transpose()), cmap='jet', interpolation='nearest')


