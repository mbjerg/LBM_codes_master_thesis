#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 09:58:34 2020

@author: bjerg
"""

# import libraries
import numpy as np
import matplotlib.pyplot as plt
from numba import njit #, prange
import time
from copy import deepcopy
#from numpy import linalg as LA
import pickle
import sys

# import own common solver functions
sys.path.insert(1,'/Users/bjerg/OneDrive - Danmarks Tekniske Universitet/master_thesis/code/common')
#from commonFuncs import collide

# set True to save result in a pickle
saveRun = False



ReTarg = 389 # target Reynolds number

NH = 30 # 7umber of grid points across channel 

AR = 20

S = int(0.94*NH)
# control parameters
Ny     = NH + S


Nx     = AR*NH         # g

max_steps = int(100e3)      # maximum number of steps in time
nu        = 0.01    # kinematic viscosity
U  = ReTarg*nu/(2*NH)          # lid velocity
Q         = 9       # number of lattice directions
cs2 = 1/3           # speed of sound ^2 
cs4 = cs2**2        # speed of sound ^4
Nout = 500          # log to screen every...
tol = 1e-9          # steady-state tolerance


inaxis = np.arange(-NH/2,NH/2) + 0.5
inlet_prof = 3/2*U*(1-(2*inaxis/NH)**2) 

# grid
#Y, X = np.meshgrid(np.arange(1,Ny+1),np.arange(1,Nx+1))
Y, X = np.meshgrid(np.linspace(0.5,Ny-0.5,Ny),np.linspace(0.5,Nx-0.5,Nx))

tau         = nu/cs2 + 1/2 
omega       = 1/tau
omega_prime = 1 - omega

# compute and display Reynolds number
Re = np.abs(2*NH*U/nu)
print(f'Reynolds number: {Re:.0f}')

# allocate space for macroscopic variables and populations
rho  = np.ones((Nx,Ny))      # macroscopic density
ux   = np.zeros((Nx,Ny))     # velocity component x-dir.
uy   = np.zeros((Nx,Ny))    # velocity component y-dir.
feq  = np.zeros((Nx,Ny,Q))   # equilibrium distribution function
fin    = np.zeros((Nx,Ny,Q))   # distribution function
fout    = np.zeros((Nx,Ny,Q))   # distribution function

# get linear indices such that Njit can loop over matrix with one for-loop
idx,jdx = np.unravel_index(np.arange(0,Nx*Ny),(Nx,Ny))



meanKinOld = 1
uxprofOld = np.ones(Ny)


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


ux[:S,:S] = np.nan
uy[:S,:S] = np.nan 
fin[:S,:S,:] = np.nan
#fin[-1,1] = 999

# weigths and velocity sets
w  = np.array([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36]) # note that last entry is the resting latice site 0
cx = np.array([0,1,0,-1,0,1,-1,-1,1])
cy = np.array([0,0,1,0,-1,1,1,-1,-1])
#flip = [3,4,1,2,7,8,5,6]
    




@njit(parallel=True)
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
        # ///  streaming  ///
    f[:,:,1] = np.roll(f[:,:,1],[+1,+0],axis=(0,1))  
    f[:,:,2] = np.roll(f[:,:,2],[+0,+1],axis=(0,1))
    f[:,:,3] = np.roll(f[:,:,3],[-1,+0],axis=(0,1))
    f[:,:,4] = np.roll(f[:,:,4],[+0,-1],axis=(0,1))
    f[:,:,5] = np.roll(f[:,:,5],[+1,+1],axis=(0,1))
    f[:,:,6] = np.roll(f[:,:,6],[-1,+1],axis=(0,1))
    f[:,:,7] = np.roll(f[:,:,7],[-1,-1],axis=(0,1))
    f[:,:,8] = np.roll(f[:,:,8],[+1,-1],axis=(0,1))
    
    return f




# ////// main loop //////
start_time = time.time()
for step in range(1,max_steps):
    
    
    # ///  collision  ///
    fout, u2 = collide(fin,feq,ux,uy,rho,omega,omega_prime)
    
    
    # /// streaming ///
    #fin = streamNjit(fin,fout,idx,jdx)
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
    #fin[-1,:,7] = fin[-2,:,7]
    #fin[-1,:,3] = fin[-2,:,3]
    #fin[-1,:,6] = fin[-2,:,6]
    
    
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
    
    # fix corners at inlet
    #fin[0,-1,5] =  fout[0,-1,7]
    #fin[0,-1,7] =  fout[0,-1,5]
    
    
    fin[:S,:S,:] = np.nan
    
    
    
    
       
    # /// update macro. quant. ///
    rho = np.sum(fin,2)
    ux  = (np.sum(fin[:,:,[1,5,8]],2) - np.sum(fin[:,:,[3,6,7]],2))/rho
    uy  = (np.sum(fin[:,:,[2,5,6]],2) - np.sum(fin[:,:,[4,7,8]],2))/rho

            
    # log to screen
    if np.mod(step,Nout) == 0:
        
        meanKin = np.mean(u2[~np.isnan(u2)]/2)   # compute mean kinetic energy (rho = 1)
        err = np.abs(meanKin-meanKinOld)/Nout/meanKinOld # compute relative error
        print(f'Step {step}, Elaps. time: {(time.time() - start_time):.2f}, Err: {err:.6f}')
        if  err < tol:
            break
        else:
            meanKinOld = meanKin
            plt.imshow(np.flipud(np.sqrt(u2).transpose()), cmap='jet', interpolation='nearest')
            plt.show()
   

#u2[:NH,:NH] = np.nan
#ux[:NH,:NH] = np.nan
#uy[:NH,:NH] = np.nan
         
print('/// DONE ///') # end the simulation

# save simulation result as pickle in ./data
if saveRun:
    filename = f'./data/Re{Re:.0f}_{Nx:.0f}x{Ny:.0f}.pickle'    
    with open(filename,'wb') as f:
        pickle.dump([X,Y,U,rho,ux,uy,Nx,Ny,NH,S,step],f)
        
    
 
plt.imshow(np.flipud(np.sqrt(u2).transpose()), cmap='jet', interpolation='nearest')

#plt.savefig('./bfs.png', bbox_inches='tight',dpi=300)
#plt.imshow(np.sqrt(u2).transpose(), cmap='hot', interpolation='nearest')
#plt.colorbar()
#plt.figure
#plt.plot(ux[0,:]/U+0,Y[0,:],'b-')
#plt.plot(ux[10,:]/U+10,Y[10,:],'b-')
#plt.plot(ux[20,:]/U+20,Y[20,:],'b-')
#plt.plot(ux[30,:]/U+30,Y[30,:],'b-')
#plt.plot(ux[40,:]/U+40,Y[40,:],'b-')
#plt.plot(ux[50,:]/U+50,Y[50,:],'b-')
#plt.plot(ux[60,:]/U+60,Y[60,:],'b-')


#plt.plot(ux[400,:]/U,Y[400,:]/NH,'b-')




#fig = plt.figure()
#
#ax = fig.add_subplot(111)
#plt.plot(ux[120,:],Y[120,:]/NH,'b-')
#ax.set_aspect(1/8)
#plt.savefig('./bfs.png', bbox_inches='tight',dpi=300)

data = np.genfromtxt('data3_57.csv',delimiter=',')





fig = plt.figure()

ax = fig.add_subplot(111)
ax.set_aspect(1)
plt.plot(data[:,0],data[:,1],'k.')
plt.plot(ux[int(3.57*S+S),:]/(U*3/2),(Y[int(3.57*S+S),:]-S)/NH,'b-') # check y-scaling



