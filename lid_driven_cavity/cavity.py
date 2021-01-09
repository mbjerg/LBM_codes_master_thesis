#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lattice Boltzmann Method CFD
Flow case: Lid Driven Cavity
Author: Morten bjerg

"""

# import libraries
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import time
import pickle
from copy import deepcopy

# set True to save results in a pickle
saveRun = False

# control parameters (lattice units)
Ny        = 101
Nx        = Ny         # number of grid points in both x- and y-direction
U         = -0.1       # lid velocity
max_steps = int(200e3) # maximum number of steps
nu        = 0.01       # kinematic viscosity
Q         = 9          # number of lattice directions
cs2       = 1/3        # speed of sound ^2
cs4       = cs2**2     # speed of sound ^4
Nout      = 500        # log to screen every Nout step
tol       = 1e-6       # steady-state convergence tolerance

# grid
Y, X = np.meshgrid(np.linspace(0.5,Ny-0.5,Ny),np.linspace(0.5,Nx-0.5,Nx))

# relaxation rate
tau         = nu/cs2 + 1/2
omega       = 1/tau
omega_prime = 1 - omega

# compute and display Reynolds number
Re = np.abs(Ny*U/nu)
print(f'Reynolds number: {Re:.0f}')

# allocate space for macroscopic variables and populations
rho  = np.ones((Nx,Ny))       # macroscopic density
ux   = np.zeros((Nx,Ny))      # velocity component x-dir.
uy   = np.zeros((Nx,Ny))      # velocity component y-dir.
feq  = np.zeros((Nx,Ny,Q))    # equilibrium distribution function
fin  = np.zeros((Nx,Ny,Q))    # distribution function
fout = np.zeros((Nx,Ny,Q))    # distribution function


# testing initialization
fin[:,:,0] = 4/9
fin[:,:,1] = 1/9
fin[:,:,2] = 1/9
fin[:,:,3] = 1/9
fin[:,:,4] = 1/9
fin[:,:,5] = 1/36
fin[:,:,6] = 1/36
fin[:,:,7] = 1/36
fin[:,:,8] = 1/36

# weigths and velocity sets
w  = np.array([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36]) 
cx = np.array([0,1,0,-1,0,1,-1,-1,1])
cy = np.array([0,0,1,0,-1,1,1,-1,-1])

# jit for extra speed
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

# -- main loop --
for step in range(1,max_steps):

    # ///  collision  ///
    fout, u2 = collide(fin,feq,ux,uy,rho,omega,omega_prime)

    # /// streaming ///
    fin = stream(deepcopy(fout))
      
    # ///  boundary conditions  ///
    # BB moving lid
    fin[:,-1,8] = fout[:,-1,6] - 2*w[6]*rho[:,-1]*np.dot([cx[6],cy[6]],[U,0])/cs2
    fin[:,-1,4] = fout[:,-1,2] - 2*w[2]*rho[:,-1]*np.dot([cx[2],cy[2]],[U,0])/cs2
    fin[:,-1,7] = fout[:,-1,5] - 2*w[5]*rho[:,-1]*np.dot([cx[5],cy[5]],[U,0])/cs2

    # BB bottom wall
    fin[:,0,6] = fout[:,0,8]
    fin[:,0,2] = fout[:,0,4]
    fin[:,0,5] = fout[:,0,7]

    # BB left wall
    fin[0,:,5] = fout[0,:,7]
    fin[0,:,1] = fout[0,:,3]
    fin[0,:,8] = fout[0,:,6]

    # BB right wall
    fin[-1,:,7] = fout[-1,:,5]
    fin[-1,:,3] = fout[-1,:,1]
    fin[-1,:,6] = fout[-1,:,8]

    # /// update macro. quant. ///
    rho = np.sum(fin,2)
    ux  = (np.sum(fin[:,:,[1,5,8]],2) - np.sum(fin[:,:,[3,6,7]],2))/rho
    uy  = (np.sum(fin[:,:,[2,5,6]],2) - np.sum(fin[:,:,[4,7,8]],2))/rho

    # log to screen
    if np.mod(step,Nout) == 0:

        meanKin = np.mean(u2/2)   # compute mean kinetic energy (rho = 1)
        err = np.abs(meanKin-meanKinOld)/Nout/meanKinOld # compute relative error
        print(f'Step {step}, Elaps. time: {(time.time() - start_time):.2f}, Err: {err:.6f}')
        if  err < tol:
            break
        else:
            meanKinOld = meanKin

print('/// DONE ///') # end the simulation


# plot
plt.figure()
plt.imshow(np.sqrt(ux**2 + uy**2))

# save simulation result as pickle in ./data
if saveRun:
    filename = f'./data/Re{Re:.0f}_{Nx:.0f}x{Ny:.0f}.pickle'
    with open(filename,'wb') as f:
        pickle.dump([X,Y,U,rho,ux,uy,Nx,Ny,step],f)
