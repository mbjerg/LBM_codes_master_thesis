#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 12:18:14 2020

@author: morten
"""

def gpu_src_code(replace_dict):
    '''
    info about function
    '''
    
    src = """
    #include <stdio.h>
    #include <math.h> 
    
    __global__ void apply_bc(int *bcidx, int *bcidy, int *bctype, float *ux, float *uy, float *rho, float *bcvaluex, float *bcvaluey)
    {
     int i = blockIdx.x * blockDim.x + threadIdx.x;
     
     if (i < Nbc) { 
     int ix = bcidx[i];
     int iy = bcidy[i];
     int ibc = iy + ix*Ny;
     //printf("%f\\n",bcvaluex[i]);
    
     
     int typ = bctype[i];
     
     if (typ == 1) {
        ux[ibc] = bcvaluex[i]; 
        uy[ibc] = bcvaluey[i];
        int ibc1 = (iy - 1) + ix*Ny;
        int ibc2 = (iy - 2) + ix*Ny;
        rho[ibc] = (4*rho[ibc1] - rho[ibc2])/3;
     }
     else if (typ == 2) {
        ux[ibc] = bcvaluex[i]; 
        uy[ibc] = bcvaluey[i];
        int ibc1 = (iy + 1) + ix*Ny;
        int ibc2 = (iy + 2) + ix*Ny;
        rho[ibc] = (4*rho[ibc1] - rho[ibc2])/3;
     }
     else if (typ == 3) {
        ux[ibc] = bcvaluex[i]; 
        uy[ibc] = bcvaluey[i];
        int ibc1 = iy + (ix + 1)*Ny;
        int ibc2 = iy + (ix + 2)*Ny;
        rho[ibc] = (4*rho[ibc1] - rho[ibc2])/3;
     }
     else if (typ == 4) {
        rho[ibc] = 1; // work-around to specify density at wall
        int ibc1 = iy + (ix - 1)*Ny;
        int ibc2 = iy + (ix - 2)*Ny;
        ux[ibc] = (4*ux[ibc1] - ux[ibc2])/3;
        uy[ibc] = (4*uy[ibc1] - uy[ibc2])/3;
     }
     else {
        printf("No such bc\\n");
     }
      
     //printf("(%d,%d); type: %d, ux: %f\\n",bcidx[i],bcidy[i],bctype[i],ux[iglob]);
     };
    
    }
     
    __global__ void predict_opt(float *ux_old, float *uy_old, float *rho_old,float *ux, float *uy, float *rho,float *ux_s, float *uy_s, float *rho_s, int *cx, int *cy, float *w)    
    {
    
      // get thread indices
      int ix = blockIdx.x * blockDim.x + threadIdx.x;
      int iy = blockIdx.y * blockDim.y + threadIdx.y;
        
      // assure that position is in the bulk flow
      if ((Nx-1) > ix && ix > 0 && (Ny-1) > iy && iy > 0) {
      
      // compute linear index
      int i = iy + ix*Ny;
      
      ux_old[i] = ux[i];
      uy_old[i] = uy[i];
      rho_old[i] = rho[i];
      
      // initialize variables as zeros
      //ux_s[i] = 0;
      //uy_s[i] = 0;
      //rho_s[i] = 0;
      
      // loop over neighbouring elements
      float ux_tmp = 0;
      float uy_tmp = 0;
      float rho_tmp = 0;
    
      for (int q = 0; q < 9; ++q) {
        int xp = ix - cx[q]; 
        int yp = iy - cy[q];
        int ip = yp + xp*Ny;
        
        float cu = 3*(cx[q]*ux[ip] + cy[q]*uy[ip]);
        float feq = rho[ip]*w[q]*(1 +  cu + 0.5*(cu*cu) - 1.5*(ux[ip]*ux[ip] + uy[ip]*uy[ip])); 
        ux_tmp += cx[q]*feq;
        uy_tmp += cy[q]*feq;
        rho_tmp += feq;
      }
      rho_s[i] = rho_tmp;
      ux_s[i] = ux_tmp/rho_tmp;
      uy_s[i] = uy_tmp/rho_tmp;
        
      };
    }
      
    __global__ void correct_opt(float *Fx, float *Fy, float *ux_old, float *uy_old, float *rho_old,float *ux, float *uy, float *rho,float *ux_s, float *uy_s, float *rho_s, int *cx, int *cy, float *w)    
    {
      // get thread indices
      int ix = blockIdx.x * blockDim.x + threadIdx.x;
      int iy = blockIdx.y * blockDim.y + threadIdx.y;
        
      // assure that position is in the bulk flow
      if ((Nx-1) > ix && ix > 0 && (Ny-1) > iy && iy > 0) {
      
      // compute linear index
      int i = iy + ix*Ny;
      
      // initialize variables as zeros
      float rho_tmp = rho_s[i];
      float ux_tmp = ux_s[i]*rho_tmp/tau_prime;
      float uy_tmp = uy_s[i]*rho_tmp/tau_prime;
      
      
      // loop over neighbouring elements
      for (int q = 0; q < 9; ++q) {
        int xp = ix + cx[q]; 
        int yp = iy + cy[q];
        int ip = yp + xp*Ny;
        
        
        float cu = 3*(cx[q]*ux_s[ip] + cy[q]*uy_s[ip]);
        float feq = rho_s[ip]*w[q]*(1 +  cu + 0.5*(cu*cu) - 1.5*(ux_s[ip]*ux_s[ip] + uy_s[ip]*uy_s[ip])); 
        
        ux_tmp += cx[q]*feq;
        uy_tmp += cy[q]*feq;
      }
       
      ux[i] = tau_prime*(ux_tmp - rho_old[i]*ux_old[i])/rho_tmp + Fx[i]/rho_tmp;
      uy[i] = tau_prime*(uy_tmp - rho_old[i]*uy_old[i])/rho_tmp + Fy[i]/rho_tmp;
      rho[i] = rho_tmp;
      
      // printf("(%d,%d); %f \\n",ix,iy,a[i]);
      
      };
    }  
      
    __global__ void Fupdate(float *ux, float *uy,float *Fx, float *Fy,float *Posx, float *Posy,float *Px, float *Py,float *Vbx, float *Vby)    
    {
    
      // get thread indices
      int ix = blockIdx.x * blockDim.x + threadIdx.x;
      int iy = blockIdx.y * blockDim.y + threadIdx.y;
      
      // assign one thread to save the velocity at blade positions
      if (ix == 0 && iy == 0) {
      
              // sketchy
              for (int b = 0; b < Nblades; ++b) {
              int ii = Posy[b] + Posx[b]*Ny;
              Vbx[b] = ux[ii];
              Vby[b] = uy[ii];
              }
      }  
      
      // assure that position is in the bulk flow
      if ((Nx-1) > ix && ix > 0 && (Ny-1) > iy && iy > 0) {
      
      // compute linear index
      int i = iy + ix*Ny;
      
      float Fx_tmp = 0.0;
      float Fy_tmp = 0.0;
      for (int b = 0; b < Nblades; ++b) {  
      float r = sqrt(pow(ix - Posx[b],2) + pow(iy - Posy[b],2));
      

      float gauss = 1/(PI*pow(chord,2))*exp(-pow(r/chord,2));
      Fx_tmp += Px[b]*gauss;
      Fy_tmp += Py[b]*gauss;
      }
      
      Fx[i] = Fx_tmp;
      Fy[i] = Fy_tmp;
      };
    }
      
    """

    for key, value in replace_dict.items():
        src = src.replace(key,str(value))
        
    return src
    