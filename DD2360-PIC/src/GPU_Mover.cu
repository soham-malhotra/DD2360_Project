#include "GPU_Mover.h"
#include <assert.h>

#define THREAD_NR 16.0

void gpu_mover_PC(struct GPUParticles** gpu_part, struct GPUEMfield* gpu_field, struct GPUGrid* gpu_grd, struct particles** part, struct parameters* param) {

    for (int is=0; is < param->ns; is++){

    int blockSize = THREAD_NR;
    int gridSize = ceil((*part)[is].nop / blockSize);

        for (int i_sub=0; i_sub < (*part)[is].n_sub_cycles; i_sub++) {
            mover_PC_kernel<<<gridSize, blockSize>>>(gpu_part[is], gpu_field, gpu_grd, *param);  // param sent to constant memory. is this worth it? it's being transferred from host to device every loop. grid is too big to save
            cudaDeviceSynchronize();
        }
    }
}  // make different species run in parallel? subcycles themselves for the same species have to be sequential

__global__ void mover_PC_kernel(struct GPUParticles* gpu_part, struct GPUEMfield* gpu_field, struct GPUGrid* gpu_grd, __grid_constant__ const struct parameters param) {

    FPpart dt_sub_cycling = (FPpart) param.dt/((double) gpu_part->n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling, qomdt2 = gpu_part->qom*dto2/param.c;  // TODO these are species specific -> move outside kernel?
    FPpart omdtsq, denom, ut, vt, wt, udotb;

    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;

    int ix,iy,iz;
    FPfield weight[8];
    FPfield xi[2], eta[2], zeta[2];  // these *should* be stored on registers, but it is the compiler that decides. otherwise, local(-global) memory

    FPpart xinit, yinit, zinit, xptilde, yptilde, zptilde, uinit, vinit, winit, uptilde, vptilde, wptilde;  // stored the approximation of x^n + 1/2
    // explictly saving initial position and velocity as automatic variables. might be too many registers + benefit compared to relying on caching can be negligible?

    int part_ind = blockIdx.x * blockDim.x + threadIdx.x;

    xinit = gpu_part->x[part_ind];
    yinit = gpu_part->y[part_ind];
    zinit = gpu_part->z[part_ind];

    uinit = gpu_part->u[part_ind];
    vinit = gpu_part->v[part_ind];
    winit = gpu_part->w[part_ind];  // really a lot of stuff to store in registers

    xptilde = xinit;
    yptilde = yinit;
    zptilde = zinit;  // initial guess for x at n + 1/2

    for(int innter=0; innter < gpu_part->NiterMover; innter++) {
        ix = 2 + int((xptilde - gpu_grd->xStart)*gpu_grd->invdx);
        iy = 2 + int((yptilde - gpu_grd->yStart)*gpu_grd->invdy);
        iz = 2 + int((zptilde - gpu_grd->zStart)*gpu_grd->invdz);  // so much indirection. is this expensive?

        int bottom_right_ind = ix * gpu_grd->nzn * gpu_grd->nyn + iy * gpu_grd->nzn + iz;
        int top_left_ind = (ix - 1) * gpu_grd->nzn * gpu_grd->nyn + (iy - 1) * gpu_grd->nzn + (iz - 1);  // avoid repeatedly re-accessing the same memory

        xi[0] = xptilde - gpu_grd->XN_GPU_flat[top_left_ind];
        eta[0] = yptilde - gpu_grd->YN_GPU_flat[top_left_ind];
        zeta[0] = zptilde - gpu_grd->ZN_GPU_flat[top_left_ind];
        xi[1] = gpu_grd->XN_GPU_flat[bottom_right_ind] - xptilde;
        eta[1] = gpu_grd->YN_GPU_flat[bottom_right_ind] - yptilde;
        zeta[1] = gpu_grd->ZN_GPU_flat[bottom_right_ind] - zptilde;
        
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                        weight[ii * 4 + jj * 2 + kk] = xi[ii] * eta[jj] * zeta[kk] * gpu_grd->invVOL;

        Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;

        for (int ii=0; ii < 2; ii++)
            for (int jj=0; jj < 2; jj++)
                for(int kk=0; kk < 2; kk++){
                    int field_ind = (ix - ii) * gpu_grd->nzn * gpu_grd->nyn + (iy - jj) * gpu_grd->nzn + (iz - kk);
                    int weight_ind = ii * 4 + jj * 2 + kk;

                    Exl += weight[weight_ind]*gpu_field->Ex_flat[field_ind];
                    Eyl += weight[weight_ind]*gpu_field->Ey_flat[field_ind];
                    Ezl += weight[weight_ind]*gpu_field->Ez_flat[field_ind];
                    Bxl += weight[weight_ind]*gpu_field->Bxn_flat[field_ind];
                    Byl += weight[weight_ind]*gpu_field->Byn_flat[field_ind];
                    Bzl += weight[weight_ind]*gpu_field->Bzn_flat[field_ind];

                }
        
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

    if (xptilde > gpu_grd->Lx){  // control divergence!
        if (param.PERIODICX==true){ // PERIODIC
            xptilde = xptilde - gpu_grd->Lx;
        } else { // REFLECTING BC
            uptilde = -uptilde;
            xptilde = 2*gpu_grd->Lx - xptilde;
        }
    }
                                                                
    if (xptilde < 0){
        if (param.PERIODICX==true){ // PERIODIC
            xptilde = xptilde + gpu_grd->Lx;
        } else { // REFLECTING BC
            uptilde = -uptilde;
            xptilde = -xptilde;
        }
    }
        
    
    // Y-DIRECTION: BC particles
    if (yptilde > gpu_grd->Ly){
        if (param.PERIODICY==true){ // PERIODIC
            yptilde = yptilde - gpu_grd->Ly;
        } else { // REFLECTING BC
            vptilde = -vptilde;
            yptilde = 2*gpu_grd->Ly - yptilde;
        }
    }
                                                                
    if (yptilde < 0){
        if (param.PERIODICY==true){ // PERIODIC
            yptilde = yptilde + gpu_grd->Ly;
        } else { // REFLECTING BC
            vptilde = -vptilde;
            yptilde = -yptilde;
        }
    }
                                                                
    // Z-DIRECTION: BC particles
    if (zptilde > gpu_grd->Lz){
        if (param.PERIODICZ==true){ // PERIODIC
            zptilde = zptilde - gpu_grd->Lz;
        } else { // REFLECTING BC
            wptilde = -wptilde;
            zptilde = 2*gpu_grd->Lz - zptilde;
        }
    }
                                                                
    if (zptilde < 0){
        if (param.PERIODICZ==true){ // PERIODIC
            zptilde = zptilde + gpu_grd->Lz;
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