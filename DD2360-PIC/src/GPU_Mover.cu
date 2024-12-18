#include "GPU_Mover.h"

#define THREAD_NR 16

void gpu_mover_PC(struct GPUParticles** gpu_part, struct GPUEMfield& gpu_field, struct GPUGrid& gpu_grd, struct particles** part, struct parameters& param) {
    //create variables for streams and gridSize 
    cudaStream_t streams[param.ns];
    int gridSize = 0;

    //create streams
    for(int is = 0; is < param.ns; is++) {
        cudaStreamCreate(&streams[is]);
    }

    //launch kernels
    for(int is = 0; is < param.ns; is++) {
        // +1 to make sure we don't miss any particles, avoid using ceil as double computation
        gridSize = ((*part)[is].nop / THREAD_NR) + 1;
        FPpart dt_sub_cycling = (FPpart) param.dt/((double) gpu_part->n_sub_cycles);
        FPpart dto2 = .5*dt_sub_cycling, qomdt2 = gpu_part->qom*dto2/param.c;  

        //run subcycles
        for (int i_sub=0; i_sub < (*part)[is].n_sub_cycles; i_sub++) {
            mover_PC_kernel<<<gridSize, THREAD_NR, 0, streams[is]>>>(gpu_part[is], 
            gpu_field, 
            gpu_grd, 
            param, 
            dto2,
            qomdt2,
            dt_sub_cycling);  
            cudaDeviceSynchronize();
        }
    }

    //destroy streams
    for(int is = 0; is < param.ns; is++) {
        cudaStreamDestroy(streams[is]);
    }
}

__global__ void mover_PC_kernel(struct GPUParticles* gpu_part, struct GPUEMfield& gpu_field, struct GPUGrid& gpu_grd, 
__grid_constant__ const struct parameters param, FPpart dto2, FPpart qomdt2, FPpart dt_sub_cycling) {

    long part_ind = blockIdx.x * blockDim.x + threadIdx.x;

    if (part_ind < gpu_part->nop) {

        FPpart x = gpu_part->x[part_ind];
        FPpart y = gpu_part->y[part_ind];
        FPpart z = gpu_part->z[part_ind];
        FPpart u = gpu_part->u[part_ind];
        FPpart v = gpu_part->v[part_ind];
        FPpart w = gpu_part->w[part_ind];
        
        FPpart xptilde = x, yptilde = y, zptilde = z;
        FPpart uptilde, vptilde, wptilde;
        FPpart omdtsq;
        FPfield weight;
        FPfield xi, eta, zeta;
        FPfield Ex=0.0, Ey=0.0, Ez=0.0, Bx=0.0, By=0.0, Bz=0.0;
        FPfield invVol = gpu_grd.invVOL;
        int ix, iy, iz;

        for(int innter = 0; innter < gpu_part->NiterMover; innter++) {
            ix = 2 + int((xptilde - gpu_grd.xStart)*gpu_grd.invdx);
            iy = 2 + int((yptilde - gpu_grd.yStart)*gpu_grd.invdy);
            iz = 2 + int((zptilde - gpu_grd.zStart)*gpu_grd.invdz);

            int base_ind = ix * gpu_grd.nzn * gpu_grd.nyn + iy * gpu_grd.nzn + iz;
            for (int ii = 0; ii < 2; ii++) {
                for (int jj = 0; jj < 2; jj++) {
                    for (int kk = 0; kk < 2; kk++) {
                        int offset = (ix - ii) * gpu_grd.nzn * gpu_grd.nyn + (iy - jj) * gpu_grd.nzn + (iz - kk);
                        FPfield weight = (xi * (ii == 0) + (1 - xi) * (ii == 1)) *
                                         (eta * (jj == 0) + (1 - eta) * (jj == 1)) *
                                         (zeta * (kk == 0) + (1 - zeta) * (kk == 1)) * invVOL;

                        Ex += weight * gpu_field.Ex_flat[offset];
                        Ey += weight * gpu_field.Ey_flat[offset];
                        Ez += weight * gpu_field.Ez_flat[offset];
                        Bx += weight * gpu_field.Bxn_flat[offset];
                        By += weight * gpu_field.Byn_flat[offset];
                        Bz += weight * gpu_field.Bzn_flat[offset];
                    }
                }
            }

            omdtsq = qomdt2*qomdt2*(Bx*Bx+By*By+Bz*Bz);
            ut = u + qomdt2 * Exl;
            vt = v + qomdt2 * Eyl;
            wt = w + qomdt2 * Ezl;
            udotb = ut * Bx + vt * By + wt * Bz;

            uptilde = (ut+qomdt2*(vt*Bz -wt*By + qomdt2*udotb*Bx))*(1.0 / (1.0 + omdtsq));  // velocity guess at n + 1/2
            vptilde = (vt+qomdt2*(wt*Bx -ut*Bz + qomdt2*udotb*By))*(1.0 / (1.0 + omdtsq));
            wptilde = (wt+qomdt2*(ut*By -vt*Bx + qomdt2*udotb*Bz))*(1.0 / (1.0 + omdtsq));

            xptilde = x + uptilde*dto2;  // position guess at n + 1/2
            yptilde = y + vptilde*dto2;
            zptilde = z + wptilde*dto2;
        }

        xptilde = x + uptilde * dt_sub_cycling;  // final positions
        yptilde = y + vptilde * dt_sub_cycling;
        zptilde = z + wptilde * dt_sub_cycling;

        uptilde = 2.0*uptilde - u;  // final velocities, no longer at 1/2 time
        vptilde = 2.0*vptilde - v;
        wptilde = 2.0*wptilde - w;

        if (xptilde > gpu_grd.Lx){  // control divergence!
            if (param.PERIODICX==true){ // PERIODIC
                xptilde = xptilde - gpu_grd.Lx;
            } else { // REFLECTING BC
                uptilde = -uptilde;
                xptilde = 2*gpu_grd.Lx - xptilde;
            }
        }
                                                                    
        if (xptilde < 0){
            if (param.PERIODICX==true){ // PERIODIC
                xptilde = xptilde + gpu_grd.Lx;
            } else { // REFLECTING BC
                uptilde = -uptilde;
                xptilde = -xptilde;
            }
        }
            
        
        // Y-DIRECTION: BC particles
        if (yptilde > gpu_grd.Ly){
            if (param.PERIODICY==true){ // PERIODIC
                yptilde = yptilde - gpu_grd.Ly;
            } else { // REFLECTING BC
                vptilde = -vptilde;
                yptilde = 2*gpu_grd.Ly - yptilde;
            }
        }
                                                                    
        if (yptilde < 0){
            if (param.PERIODICY==true){ // PERIODIC
                yptilde = yptilde + gpu_grd.Ly;
            } else { // REFLECTING BC
                vptilde = -vptilde;
                yptilde = -yptilde;
            }
        }
                                                                    
        // Z-DIRECTION: BC particles
        if (zptilde > gpu_grd.Lz){
            if (param.PERIODICZ==true){ // PERIODIC
                zptilde = zptilde - gpu_grd.Lz;
            } else { // REFLECTING BC
                wptilde = -wptilde;
                zptilde = 2*gpu_grd.Lz - zptilde;
            }
        }
                                                                    
        if (zptilde < 0){
            if (param.PERIODICZ==true){ // PERIODIC
                zptilde = zptilde + gpu_grd.Lz;
            } else { // REFLECTING BC
                wptilde = -wptilde;
                zptilde = -zptilde;
            }
        }

        gpu_part->u[part_ind] = uptilde;  // save final values
        gpu_part->v[part_ind] = vptilde;
        gpu_part->w[part_ind] = wptilde;

        gpu_part->x[part_ind] = xptilde;
        gpu_part->y[part_ind] = yptilde;
        gpu_part->z[part_ind] = zptilde;

        }
        /*
            omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
            denom = 1.0/(1.0 + omdtsq);

            ut = uinit + qomdt2 * Exl;
            vt = vinit + qomdt2 * Eyl;
            wt = winit + qomdt2 * Ezl;
            udotb = ut * Bxl + vt * Byl + wt * Bzl;

            uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;  // velocity guess at n + 1/2
            vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
            wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;

            xptilde = xinit + uptilde*dto2;  // position guess at n + 1/2
            yptilde = yinit + vptilde*dto2;
            zptilde = zinit + wptilde*dto2;
        }

        xptilde = xinit + uptilde * dt_sub_cycling;  // final positions
        yptilde = yinit + vptilde * dt_sub_cycling;
        zptilde = zinit + wptilde * dt_sub_cycling;

        uptilde = 2.0*uptilde - uinit;  // final velocities, no longer at 1/2 time
        vptilde = 2.0*vptilde - vinit;
        wptilde = 2.0*wptilde - winit;

        if (xptilde > gpu_grd.Lx){  // control divergence!
            if (param.PERIODICX==true){ // PERIODIC
                xptilde = xptilde - gpu_grd.Lx;
            } else { // REFLECTING BC
                uptilde = -uptilde;
                xptilde = 2*gpu_grd.Lx - xptilde;
            }
        }
                                                                    
        if (xptilde < 0){
            if (param.PERIODICX==true){ // PERIODIC
                xptilde = xptilde + gpu_grd.Lx;
            } else { // REFLECTING BC
                uptilde = -uptilde;
                xptilde = -xptilde;
            }
        }
            
        
        // Y-DIRECTION: BC particles
        if (yptilde > gpu_grd.Ly){
            if (param.PERIODICY==true){ // PERIODIC
                yptilde = yptilde - gpu_grd.Ly;
            } else { // REFLECTING BC
                vptilde = -vptilde;
                yptilde = 2*gpu_grd.Ly - yptilde;
            }
        }
                                                                    
        if (yptilde < 0){
            if (param.PERIODICY==true){ // PERIODIC
                yptilde = yptilde + gpu_grd.Ly;
            } else { // REFLECTING BC
                vptilde = -vptilde;
                yptilde = -yptilde;
            }
        }
                                                                    
        // Z-DIRECTION: BC particles
        if (zptilde > gpu_grd.Lz){
            if (param.PERIODICZ==true){ // PERIODIC
                zptilde = zptilde - gpu_grd.Lz;
            } else { // REFLECTING BC
                wptilde = -wptilde;
                zptilde = 2*gpu_grd.Lz - zptilde;
            }
        }
                                                                    
        if (zptilde < 0){
            if (param.PERIODICZ==true){ // PERIODIC
                zptilde = zptilde + gpu_grd.Lz;
            } else { // REFLECTING BC
                wptilde = -wptilde;
                zptilde = -zptilde;
            }
        }

        gpu_part->u[part_ind] = uptilde;  // save final values
        gpu_part->v[part_ind] = vptilde;
        gpu_part->w[part_ind] = wptilde;

        gpu_part->x[part_ind] = xptilde;
        gpu_part->y[part_ind] = yptilde;
        gpu_part->z[part_ind] = zptilde;

          */
  
    }

    
}