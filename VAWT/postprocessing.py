#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 13:48:55 2020

@author: morten
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt




filename = './data/id0.pickle'
with open(filename, 'rb') as f:
    R,omega, X,Y,U,rho,ux,uy,Pt_array,Pn_array,theta_array,alpha_array,T_array,P_array = pickle.load(f)
    
data_check = np.loadtxt('./vortex/TSR_258_ct_vortex.csv',delimiter=',')    
    

rho_ref = 1
Nrev = 2*np.pi/omega

Rev_start = 3
Rev_end = 4

Nstart = int(Nrev*Rev_start)
Nend = int(Nrev*Rev_end)

 # %%   
plt.figure(1)
plt.plot(data_check[:,0],data_check[:,1],'r-')
plt.plot(theta_array[Nstart:Nend] - theta_array[Nstart],Pt_array[Nstart:Nend]/(0.5*2*R*rho_ref*U**2),'b-')
plt.xlabel('azimuthal angle')
plt.ylabel('Ct')
plt.xlim(0,360)
plt.ylim(-0.1,0.5)

# # %%
# plt.figure(2)

# plt.plot(theta_array,Pn_array,'b-')

# # %%
# plt.figure(3)

# plt.plot(theta_array,alpha_array,'b-')

# %%
umag = np.sqrt(ux**2 + uy**2)
umag_rot = np.rot90(umag)
plt.figure(4)
plt.imshow(umag_rot,cmap='jet')

# %%
Nx, Ny = X.shape

# compute the vorticity
dudy = np.zeros([Nx-2,Ny-2])
dvdx = np.zeros([Nx-2,Ny-2])

for i in range(1,Nx-1):
    for j in range(1,Ny-1):
        dudy[i-1,j-1] = (ux[i,j+1] - ux[i,j-1])/2
        
for i in range(1,Nx-1):
    for j in range(1,Ny-1):
        dvdx[i-1,j-1] = (uy[i+1,j] - uy[i-1,j])/2

vort = dvdx - dudy
vort_rot = np.rot90(vort)    
        
# %%
plt.figure(5)
plt.imshow(vort_rot,cmap = 'jet_r')
        
#plt.figure(6)
#plt.imshow(vort_rot[100:200,100:400],cmap = 'jet_r')




