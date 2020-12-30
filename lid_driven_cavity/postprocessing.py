#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 12:52:02 2020
@author: bjerg
"""
from matplotlib import pyplot as plt
import numpy as np
import pickle

# load simulation with the following parameters
Re = 1000
N = 251



# grids 31 51 91 171
# tol: 1e-6


filename = f'./data/Re{Re:.0f}_{N:.0f}x{N:.0f}.pickle'
with open(filename, 'rb') as f:
    X,Y,U,rho,ux,uy,Nx,Ny,step = pickle.load(f)


# load benchmark solutions from bruneau & saad
bmark_hoz = np.genfromtxt('./data/bruneau_saad_horizontalprofile.csv', delimiter=',')
bmark_vert = np.genfromtxt('./data/bruneau_saad_verticalprofile.csv', delimiter=',')

# speed of sound squared
cs2 = 1/3

# compute velocity mag.   
u = np.sqrt(ux**2 + uy**2) 




plt.figure(1)
plt.title('Horizontal centerline velocity profile')
plt.plot(X[:,int(Ny/2)]/Ny,uy[:,int(Ny/2)]/np.abs(U),'b-')
plt.plot(bmark_hoz[:,0],bmark_hoz[:,1],'ko')
plt.xlabel('x')
plt.ylabel('v')
plt.legend(['Present study','Bruneau & Saad'])
plt.savefig('./figures/cavity1.png', bbox_inches='tight',dpi=300)


plt.figure(2)
plt.title('Vertical centerline velocity profile')
plt.plot(ux[int(Nx/2),:]/np.abs(U),Y[int(Ny/2),:]/Ny,'b-')
plt.plot(bmark_vert[:,1],bmark_vert[:,0],'ko')
plt.xlabel('u')
plt.ylabel('y')
plt.savefig('./figures/cavity2.png', bbox_inches='tight',dpi=300)

#plt.figure(3)
#plt.title('Vertical centerline pressure')
#plt.plot((rho[int(Nx/2),:]-rho[int(Nx/2),int(Nx/2)])*cs2/(np.mean(rho)*U**2),Y[int(Ny/2),:]/Ny)
#plt.plot(bmark_vert[:,2],bmark_vert[:,0])
#plt.legend(['Present','Bruneau & Saad'])

#plt.figure(4)
#plt.title('Vertical centerline pressure')
#nor = np.max(rho[int(Nx/2),:]-rho[int(Nx/2),int(Nx/2)])*cs2
#plt.plot((rho[int(Nx/2),:]-rho[int(Nx/2),int(Nx/2)])*cs2/nor,Y[int(Ny/2),:]/Ny)
#plt.plot(bmark_vert[:,2]/np.max(bmark_vert[:,2]),bmark_vert[:,0])
#plt.legend(['Present','Bruneau & Saad'])


plt.figure(5)
plt.title('Vertical centerline pressure profile')
plt.plot((rho[int(Nx/2),:]-rho[int(Nx/2),int(Nx/2)])*cs2/U**2/np.mean(rho),Y[int(Ny/2),:]/Ny,'b-')
plt.plot(bmark_vert[:,2],bmark_vert[:,0],'ko')
plt.xlabel('p')
plt.ylabel('y')
#plt.legend(['Present study','Bruneau & Saad'])
plt.savefig('./figures/cavity3.png', bbox_inches='tight',dpi=300)


plt.figure(6)
plt.title('Horizontal centerline pressure profile')
plt.plot(X[:,int(Ny/2)]/Nx,(rho[:,int(Nx/2)]-rho[int(Nx/2),int(Nx/2)])*cs2/U**2/np.mean(rho),'b-')
plt.plot(bmark_hoz[:,0],bmark_hoz[:,2],'ko')
plt.xlabel('x')
plt.ylabel('p')
plt.savefig('./figures/cavity4.png', bbox_inches='tight',dpi=300)

#plt.figure(2)
#plt.plot((rho[int(Nx/2),:]-rho[15,15])/3,y/Ny) #plt.figure(1)

fig, ax = plt.subplots()
plt.streamplot(Y, X, uy, ux,density=2, color='b')
plt.xlabel('y')
plt.ylabel('x')
ax.set_aspect('equal')
#plt.xticks(rotation=-90)
#plt.yticks(rotation=-90)
fig.patch.set_visible(False)
ax.axis('off')
plt.show
#plt.savefig('test.png', bbox_inches='tight',dpi=300)
plt.savefig('./figures/cavity5.png', bbox_inches='tight',dpi=300)



#plt.quiver(X,Y,ux,uy)
#plt.figure(2)
#plt.imshow(u.transpose(), cmap='hot', interpolation='nearest')
#%lt.figure(2)
#plt.plot(ux[int(N/2),:]/Ulid)