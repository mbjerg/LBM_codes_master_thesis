#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 18:46:14 2020

@author: bjerg
"""
from numba import njit
import numpy as np


@njit()
def collide(f,feq,ux,uy,rho,omega,omega_prime,S):
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
    fstar = f*omega_prime + feq*omega + S

    return fstar , u2



@njit()
def getFeq(feq,ux,uy,rho):
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

    return feq




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

@njit()
def applyBC(fin,fout,w,rho,cx,cy,ux,uy,cs2,cs4):
    # BB inlet
    fin[0,:,5] = fout[-1,:,5] 
    fin[0,:,1] = fout[-1,:,1] 
    fin[0,:,8] = fout[-1,:,8] 
        
    # anti-BB outlet 
    fin[-1,:,7] = fout[0,:,7] 
    fin[-1,:,3] = fout[0,:,3] 
    fin[-1,:,6] = fout[0,:,6] 
           
    # BB top wall
    fin[:,-1,8] = fout[:,-1,6] 
    fin[:,-1,4] = fout[:,-1,2] 
    fin[:,-1,7] = fout[:,-1,5] 
    
    # BB bottom wall
    fin[:,0,6] = fout[:,0,8] 
    fin[:,0,2] = fout[:,0,4] 
    fin[:,0,5] = fout[:,0,7] 
    
    return fin

@njit()
def updateMacro(fin,Fgauss):
    rho = fin[:,:,0] + fin[:,:,1] + fin[:,:,2] + fin[:,:,3] + fin[:,:,4] + fin[:,:,5] + fin[:,:,6] + fin[:,:,7]+ fin[:,:,8]
    ux = ( fin[:,:,1] + fin[:,:,5] + fin[:,:,8] - (fin[:,:,3] + fin[:,:,6] + fin[:,:,7]) )/rho
    uy  = ( fin[:,:,2] + fin[:,:,5] + fin[:,:,6] - (fin[:,:,4] + fin[:,:,7] + fin[:,:,8]) )/rho
    
    return rho, ux, uy


def compSource(Nx,Ny,Q,tau,w,cx,ux,cs2,cs4,F):
        S = np.zeros([Nx,Ny,Q])
        for i in range(0,9):
            S[:,:,i] = (1-1/(2*tau))*w[i]*((cx[i]-ux)/cs2 + (cx[i]*ux)*cx[i]/cs4)*F
            
        return S
