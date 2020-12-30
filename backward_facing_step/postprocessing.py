#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 12:49:39 2020

@author: bjerg
"""
import pickle
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
Re =389




filename = f'./data/Re{Re:.0f}.pickle'
with open(filename, 'rb') as f:
    X,Y,U,rho,ux,uy,Nx,Ny,NH,S,step = pickle.load(f)
    
bm0 = np.genfromtxt('./data/Re389_XS0.csv', delimiter=',')   
bm1 = np.genfromtxt('./data/Re389_XS255.csv', delimiter=',')
bm2 = np.genfromtxt('./data/Re389_XS306.csv', delimiter=',')
bm3 = np.genfromtxt('./data/Re389_XS357.csv', delimiter=',')


XS = [0,2.55,3.06,3.57]
Umax = 3/2*U




fig = plt.figure()

ax = fig.add_subplot(111)
ax.set_aspect(1)
plt.plot(ux[int(XS[0]*S+S),:]/(Umax),(Y[int(XS[0]*S+S),:]-S)/NH,'b-') # check y-scaling
plt.plot(bm0[:,0],bm0[:,1],'k.')
plt.xlabel('u/Umax')
plt.ylabel('y/h')
plt.xticks([0,0.5,1])
#plt.yticks([])
plt.title(f'x/S = {XS[0]}')
plt.legend(['Present study','Armaly et al.'])
plt.savefig('./figures/bfs_prof0.png', bbox_inches='tight',dpi=300)

fig = plt.figure()

ax = fig.add_subplot(111)
ax.set_aspect(1)
plt.plot(bm1[:,0],bm1[:,1],'k.')
plt.plot(ux[int(XS[1]*S+S),:]/(Umax),(Y[int(XS[1]*S+S),:]-S)/NH,'b-') # check y-scaling
plt.xlabel('u/Umax')
#plt.ylabel('y/h')
plt.xticks([0,0.5,1])

plt.title(f'x/S = {XS[1]}')
plt.savefig('./figures/bfs_prof1.png', bbox_inches='tight',dpi=300)




fig = plt.figure()

ax = fig.add_subplot(111)
ax.set_aspect(1)
plt.plot(bm2[:,0],bm2[:,1],'k.')
plt.plot(ux[int(XS[2]*S+S),:]/(Umax),(Y[int(XS[2]*S+S),:]-S)/NH,'b-') # check y-scaling
plt.xlabel('u/Umax')
#plt.ylabel('y/h')
plt.xticks([0,0.5,1])

plt.title(f'x/S = {XS[2]}')
plt.savefig('./figures/bfs_prof2.png', bbox_inches='tight',dpi=300)




fig = plt.figure()

ax = fig.add_subplot(111)
ax.set_aspect(1)
plt.plot(bm3[:,0],bm3[:,1],'k.')
plt.plot(ux[int(XS[3]*S+S),:]/(Umax),(Y[int(XS[3]*S+S),:]-S)/NH,'b-') # check y-scaling
#plt.xlabel('$\mathregular{u/U_{max}}$')
plt.xlabel('u/Umax')#plt.ylabel('y/h')
plt.title(f'x/S = {XS[3]}')
plt.xticks([0,0.5,1])

plt.title(f'x/S = {XS[3]}')
plt.savefig('./figures/bfs_prof3.png', bbox_inches='tight',dpi=300)





# collective plot




#
#
#fig = plt.figure()
#plt.plot(bm0[:,0],bm0[:,1],'k.')
#plt.plot(ux[int(XS[0]*S+S),:]/(Umax),(Y[int(XS[0]*S+S),:]-S)/NH,'b-') # check y-scaling
#
#
#
#
#plt.plot(ux[int(XS[1]*S+S),:]/(Umax)+1,(Y[int(XS[1]*S+S),:]-S)/NH,'b-') # check y-scaling
#plt.plot(bm1[:,0]+1,bm1[:,1],'k.')
#
#plt.plot(ux[int(XS[2]*S+S),:]/(Umax)+2,(Y[int(XS[2]*S+S),:]-S)/NH,'b-') # check y-scaling
#plt.plot(bm2[:,0]+2,bm2[:,1],'k.')
#
#plt.plot(ux[int(XS[3]*S+S),:]/(Umax)+3,(Y[int(XS[3]*S+S),:]-S)/NH,'b-') # check y-scaling
#plt.plot(bm3[:,0]+3,bm3[:,1],'k.')
#




u = np.sqrt(ux**2 + uy**2)/Umax
levels = np.linspace(np.min(u[~np.isnan(u)]),np.max(u[~np.isnan(u)]),50)
fig = plt.figure(figsize=(10, 3))
ax = fig.add_subplot(111)



plt.xlabel('x/S')
plt.ylabel('y/h')
plt.yticks([0])
im = plt.contourf((X-S)/S,(Y-S)/NH,u,levels,antialiased=False,cmap='jet')
fig.colorbar(im,ticks=[np.min(u[~np.isnan(u)]),0.25,0.5,0.75,1], orientation="horizontal", pad=0.3)
#
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.set_aspect(1/4)
#plt.streamplot(Y,X,uy,ux,density=1)

#plt.savefig('./figures/bfc_Umag.png', bbox_inches='tight',dpi=300)


