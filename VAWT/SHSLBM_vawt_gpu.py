#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 11:54:09 2020
@author: morten
"""
# import modules
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
#import matplotlib.pyplot as plt
import time
from kernel_src import gpu_src_code
import pickle
import datetime

# save the end result in pickle?
jobID = 1
saveRun = True

# set target Reynolds number (for case 2, Re = 800,000)
# based on free wind, radius and kinematic viscosity of air
Re_targ = 800e3 

# /// general setup ///
Ny = int(2500) 
Nx = int(2*Ny)
N = Nx*Ny # total number of grid points
Nbc = 2*(Nx + Ny) - 4 # number of boundary grid points
U = 0.01 # wind speed (x-comp.)
Nout = int(5000) # log every Nout step...
Q = 9 # stencils D2Q9

# construct grid
Y, X = np.meshgrid(np.arange(Ny),np.arange(Nx))

# /// vawt settings ///
Nblades = 2
block = 0.10 # vawt domain blockage ratio
D = int(block*Ny) # diameter for vawt
R = D//2 # radius of vawt
c = R*0.265 # airfoil chord
TSR = 2.58 # tip-speed-ratio
omega = TSR*U/R # rotational speed
alpha_p = 6 # pitch [deg]
rho_ref = 1 # reference density
theta0 = 0 # starting position for blade with index 0
xc = X[Nx//3,Ny//2] # center of rotation, shaft position
yc = Y[Nx//3,Ny//2]

# set relaxation parameter to match the target Reynolds number
nu = U*R/Re_targ # viscosity
cs2 = 1/3 # speed of sound
tau = nu/cs2 + 1/2 # relaxation rate
tau_prime = tau - 1

# compute Reynolds number which should match the target 
Re = U*R/nu

# compute the steps needed to perform desired # of rotations
revs = 12# number of revolutions
Nrev = np.pi*2/omega
max_steps = int(revs*np.pi*2/omega) # max number of iterations

# write file with simulation configuration information
conf_dict = {
             'Simulation ID': jobID,
             'Re': int(Re),   
             'Nx': Nx,
             'Ny': Ny,
             'Uinf': U,
             'Density': rho_ref,
             'Kin. visc.': nu,
             'Blades': Nblades,
             'R': R,
             'Chord': c,
             'TSR': TSR,
             'omega': omega,
             'Pitch [deg]': alpha_p,
             'Revolutions': revs,
             'Iterations': max_steps
             }

conf_str = '---- simulation setup ----\n'
for key, value in conf_dict.items():
    conf_str = conf_str + key + ': ' + str(value) + '\n'

with open('./data/jobID_' + str(jobID) +'_specs.txt', 'w') as f:
    f.write(conf_str)


# load airfoil data
data = np.data = np.genfromtxt('./data/n0021_galih_ds.dat')
naca0021_alpha = np.deg2rad(data[:,0])
naca0021_Cl = data[:,1]
naca0021_Cd = data[:,2]

# mask out bulk flow and assign boundary conditions
mask = np.ones([Nx,Ny],np.bool)
mask[1:-1,1:-1] = False 
bcidx = X[mask].astype(np.int32)
bcidy = Y[mask].astype(np.int32)

# boundary node type which is translated to actual condition on gpu
bctype = np.zeros([Nx,Ny]).astype(np.int32)
bctype[:, -1] = 1
bctype[:, 0] = 2
bctype[0, 1:-1] = 3
bctype[-1, 1:-1] = 4
bctype = bctype[mask]

# boudary node values for dirichlet conditions
bcvaluex = np.zeros([Nx,Ny]).astype(np.float32)
bcvaluey = np.copy(bcvaluex)
bcvaluex[0,1:-1] = U
bcvaluex[:,0] = U
bcvaluex[:,-1] = U
bcvaluex = bcvaluex[mask]
bcvaluey = bcvaluey[mask]

#  /// initialize variables ///
Fx = np.zeros([Nx,Ny]).astype(np.float32) 
Fy = np.zeros([Nx,Ny]).astype(np.float32)

Vbx = np.zeros(Nblades).astype(np.float32)
Vby = np.zeros(Nblades).astype(np.float32)

Posx = np.zeros(Nblades).astype(np.float32)
Posy = np.zeros(Nblades).astype(np.float32)
Px = np.zeros(Nblades).astype(np.float32)
Py = np.zeros(Nblades).astype(np.float32)

ux = U*np.ones([Nx,Ny]).astype(np.float32)
uy = np.zeros([Nx,Ny]).astype(np.float32)
rho = np.ones([Nx,Ny]).astype(np.float32)
ux[:,-1] = U

ux_s = np.ones([Nx,Ny]).astype(np.float32)
uy_s = np.ones([Nx,Ny]).astype(np.float32)
rho_s = np.ones([Nx,Ny]).astype(np.float32)

ux_old = np.random.rand(Nx,Ny).astype(np.float32)
uy_old = np.random.rand(Nx,Ny).astype(np.float32)
rho_old = np.ones([Nx,Ny]).astype(np.float32)

cx = np.array([0,1,0,-1,0,1,-1,-1,1]).astype(np.int32)
cy = np.array([0,0,1,0,-1,1,1,-1,-1]).astype(np.int32)
w = np.array([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36]).astype(np.float32)

# /// allocate memory on gpu ///
Vbx_gpu = cuda.mem_alloc(Vbx.nbytes)
Vby_gpu = cuda.mem_alloc(Vby.nbytes)

Posx_gpu = cuda.mem_alloc(Posx.nbytes)
Posy_gpu = cuda.mem_alloc(Posy.nbytes)
Px_gpu = cuda.mem_alloc(Px.nbytes)
Py_gpu = cuda.mem_alloc(Py.nbytes)

Fx_gpu = cuda.mem_alloc(Fx.nbytes)
Fy_gpu = cuda.mem_alloc(Fx.nbytes)

bcvaluex_gpu = cuda.mem_alloc(bcvaluex.nbytes)
bcvaluey_gpu = cuda.mem_alloc(bcvaluey.nbytes)

bcidx_gpu = cuda.mem_alloc(bcidx.nbytes)
bcidy_gpu = cuda.mem_alloc(bcidy.nbytes)
bctype_gpu = cuda.mem_alloc(bctype.nbytes)

ux_gpu = cuda.mem_alloc(ux.nbytes)
uy_gpu = cuda.mem_alloc(uy.nbytes)
rho_gpu = cuda.mem_alloc(rho.nbytes)

ux_s_gpu = cuda.mem_alloc(ux_s.nbytes)
uy_s_gpu = cuda.mem_alloc(uy_s.nbytes)
rho_s_gpu = cuda.mem_alloc(rho_s.nbytes)

ux_old_gpu = cuda.mem_alloc(ux_old.nbytes)
uy_old_gpu = cuda.mem_alloc(uy_old.nbytes)
rho_old_gpu = cuda.mem_alloc(rho_old.nbytes)

cx_gpu = cuda.mem_alloc(cx.nbytes)
cy_gpu = cuda.mem_alloc(cy.nbytes)
w_gpu = cuda.mem_alloc(w.nbytes)

#%  /// transfer to gpu ///
cuda.memcpy_htod(Vbx_gpu,Vbx)
cuda.memcpy_htod(Vby_gpu,Vby)

cuda.memcpy_htod(Posx_gpu,Posx)
cuda.memcpy_htod(Posy_gpu,Posy)
cuda.memcpy_htod(Px_gpu,Px)
cuda.memcpy_htod(Py_gpu,Py)

cuda.memcpy_htod(Fx_gpu,Fx)
cuda.memcpy_htod(Fy_gpu,Fy)

cuda.memcpy_htod(bcvaluex_gpu,bcvaluex)
cuda.memcpy_htod(bcvaluey_gpu,bcvaluey)

cuda.memcpy_htod(bcidx_gpu,bcidx)
cuda.memcpy_htod(bcidy_gpu,bcidy)
cuda.memcpy_htod(bctype_gpu,bctype)

cuda.memcpy_htod(ux_gpu,ux)
cuda.memcpy_htod(uy_gpu,uy)
cuda.memcpy_htod(rho_gpu,rho)

cuda.memcpy_htod(ux_s_gpu,ux_s)
cuda.memcpy_htod(uy_s_gpu,uy_s)
cuda.memcpy_htod(rho_s_gpu,rho_s)

cuda.memcpy_htod(ux_old_gpu,ux_old)
cuda.memcpy_htod(uy_old_gpu,uy_old)
cuda.memcpy_htod(rho_old_gpu,rho_old)

cuda.memcpy_htod(w_gpu,w)
cuda.memcpy_htod(cx_gpu,cx)
cuda.memcpy_htod(cy_gpu,cy)

# dictionary of variables which must be replaced directly in the gpu src code
replace_dict = {
           'chord': c,
           'PI': np.pi,
           'Nx': Nx,
           'Ny': Ny,
           'Ntot': N,
           'Nbc': Nbc,
           'Nblades': Nblades,
           'tau_prime': tau_prime,
           }
   
# retrieve gpu src code 
module = SourceModule(gpu_src_code(replace_dict))

# initialize kernels
predict_opt = module.get_function("predict_opt")
correct_opt = module.get_function("correct_opt")
apply_bc = module.get_function("apply_bc")
Fupdate = module.get_function("Fupdate")

# decide the number of threads and blocks needed for the different kernels
Nthreads = 8
Nthreads_bc = 16
Ngridsx = (Nx + Nthreads - 1)//Nthreads
Ngridsy = (Ny + Nthreads - 1)//Nthreads
Ngridsx_bc = (Nbc + Nthreads_bc - 1)//Nthreads_bc

# run-time storage arrays for sampling
storage = np.empty(max_steps)
storage[:] = np.nan

P_array = np.copy(storage)
T_array = np.copy(storage)
Pt_array = np.copy(storage)
Pn_array = np.copy(storage)
theta_array = np.copy(storage)
alpha_array = np.copy(storage)

# start timing 
MLUPS_start = time.time()
T0 = time.time()

# attempt to predict the total simulation time in minutes
# TODO: make prediction based on recorded MLUPS instead of hardcoded value 250
MLUPS_avg = 250
Est_sim_time = Nx*Ny*max_steps/(MLUPS_avg*1e6)/60
print(f'Projected total simulation time: {Est_sim_time:.0f} min.')

# Let's go!
for i in range(max_steps):
      
    # prediction step
    predict_opt(ux_old_gpu,uy_old_gpu, rho_old_gpu, ux_gpu,uy_gpu,rho_gpu,ux_s_gpu,uy_s_gpu,rho_s_gpu,cx_gpu,cy_gpu,w_gpu,block=(Nthreads,Nthreads,1),grid=(Ngridsx,Ngridsy,1))
    
    # apply bc's again
    apply_bc(bcidx_gpu,bcidy_gpu,bctype_gpu,ux_s_gpu,uy_s_gpu,rho_s_gpu,bcvaluex_gpu,bcvaluey_gpu,block=(Nthreads_bc,1,1),grid=(Ngridsx_bc,1,1))
    
    # correction step
    correct_opt(Fx_gpu,Fy_gpu,ux_old_gpu,uy_old_gpu,rho_old_gpu,ux_gpu,uy_gpu,rho_gpu,ux_s_gpu,uy_s_gpu,rho_s_gpu,cx_gpu,cy_gpu,w_gpu,block=(Nthreads,Nthreads,1),grid=(Ngridsx,Ngridsy,1))
    
    # apply bc's again
    apply_bc(bcidx_gpu,bcidy_gpu,bctype_gpu,ux_gpu,uy_gpu,rho_gpu,bcvaluex_gpu,bcvaluey_gpu,block=(Nthreads_bc,1,1),grid=(Ngridsx_bc,1,1))
    
    # update the force field
    Fupdate(ux_gpu,uy_gpu,Fx_gpu,Fy_gpu,Posx_gpu,Posy_gpu,Px_gpu,Py_gpu,Vbx_gpu,Vby_gpu,block=(Nthreads,Nthreads,1),grid=(Ngridsx,Ngridsy,1))
      
    # update force
    theta0 += omega 
     
    # retrieve the velocity at the blades positions from gpu  
    cuda.memcpy_dtoh(Vbx,Vbx_gpu)
    cuda.memcpy_dtoh(Vby,Vby_gpu)

    # compute the body force contribution from each blade 
    for b in range(Nblades):
        
        theta = theta0 + b*(2*np.pi/Nblades)
           
        x0 = -R*np.sin(theta) + xc
        y0 =  R*np.cos(theta) + yc
        
        Posx[b] = np.round(x0) # has to be an int to work on gpu
        Posy[b] = np.round(y0)
           
        # TODO: fix the fact that the velocity fetched from the gpu is lagging
        # behind as the 'old' angular position is used for sampling  
        Vx = Vbx[b] 
        Vy = Vby[b] 
            
        Vrel_t = omega*R + Vx*np.cos(theta) + Vy*np.sin(theta)
        Vrel_n = Vx*np.sin(theta) - Vy*np.cos(theta)
        Vrel_mag = np.sqrt(Vrel_t**2 + Vrel_n**2)
        
        phi = np.arctan(Vrel_n/Vrel_t) # flow angle   
        alpha = phi - np.deg2rad(alpha_p) # angle of attack
       
        #Cl = 1.11*2*np.pi*np.sin(alpha) # hardcoded polars
        #Cd = 0 # drag coefficient
        Cl = np.interp(alpha,naca0021_alpha,naca0021_Cl)
        Cd = np.interp(alpha,naca0021_alpha,naca0021_Cd)
        
        l = 0.5*rho_ref*Vrel_mag**2*c*Cl # lift
        d = 0.5*rho_ref*Vrel_mag**2*c*Cd # drag
        
        Pt = d*np.cos(phi) - l*np.sin(phi) # tangential force     
        Pn = l*np.cos(phi) + d*np.sin(phi) # normal force
   
        # body force acting on fluid in global xy coordinate system
        Px[b] = -( Pt*np.cos(theta) + Pn*np.sin(theta) ) 
        Py[b] = -( Pt*np.sin(theta) - Pn*np.cos(theta) ) 
        
        # track blade with index 0 
        if b == 0:    
            Ptcomb = Pt
            Pt_array[i] = -Pt
            Pn_array[i] = Pn
            alpha_array[i] = np.rad2deg(alpha)
            theta_array[i] = np.rad2deg(theta)
        else:
            Ptcomb += Pt
            
    # track thrust and power       
    T_array[i] = np.sum(-Px) 
    P_array[i] = -omega*R*Ptcomb
        
    # transfer body force vector to gpu   
    cuda.memcpy_htod(Px_gpu,Px)
    cuda.memcpy_htod(Py_gpu,Py)
    cuda.memcpy_htod(Posx_gpu,Posx)
    cuda.memcpy_htod(Posy_gpu,Posy)
    
    # log to console and display velocity field
    if np.mod(i,Nout) == 0:
        MLUPS_end = time.time()
        MLUPS = Nout*Nx*Ny/(MLUPS_end - MLUPS_start)*1e-6
        Time = str(datetime.timedelta(seconds = np.round(time.time() - T0)))
        
        # flush = True needed to force clust to print during run
        print(f'Iter.: {i}, T:{Time}s, MLUPS: {MLUPS:.2f}', flush = True)
        #cuda.memcpy_dtoh(ux,ux_gpu)
        #cuda.memcpy_dtoh(uy,uy_gpu)
        #cuda.memcpy_dtoh(rho,rho_gpu)
        
        # plot velocity field
        # plt.figure(num=None, figsize=(6, 4), dpi=150, facecolor='w', edgecolor='k')
        # plt.imshow(np.sqrt(ux**2 + uy**2),cmap='jet') #vmin=U*0.88, vmax=U*1.05
        # plt.colorbar()
        # plt.show()
        
        MLUPS_start = time.time()
        
#  end of run        
print("/// Done ///")

# get final velocity field from gpu
cuda.memcpy_dtoh(ux,ux_gpu)
cuda.memcpy_dtoh(uy,uy_gpu)
cuda.memcpy_dtoh(rho,rho_gpu)
        
# %% save simulation result as pickle in ./data
if saveRun:
    filename = f'./data/id{jobID}.pickle'    
    with open(filename,'wb') as f:
        pickle.dump([R,omega,X,Y,U,rho,ux,uy,Pt_array,Pn_array,theta_array,alpha_array,T_array,P_array],f)

# Nfrom = (revs-3)*int(Nrev)
# Nto = (revs-2)*int(Nrev)

# plt.figure()
# plt.plot(np.rad2deg(theta_array[Nfrom:Nto]),-Pt_array[Nfrom:Nto]/(0.5*rho_ref*U**2*2*R),'b-')
# plt.xlabel('azimuthal angle')
# plt.ylabel('Ct')

# plt.figure()
# plt.plot(np.rad2deg(theta_array),Pn_array/(0.5*rho_ref*U**2*2*R),'b-')
# plt.xlabel('azimuthal angle')
# plt.ylabel('Cn')


#plt.figure()
#plt.plot(x_array,y_array)
#plt.gca().set_aspect('equal', adjustable='box')
