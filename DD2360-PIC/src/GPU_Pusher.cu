#include "GPU_Parameters.h"
#include "GPU_Grid.h"
#include "GPU_EMfield.h"
#include "GPU_Particles.h"

#include "GPU_Pusher.h"

#include <cuda.h>
#include <cuda_runtime.h>

#define THREAD_NR 16.0

void particle_subcycler(struct GPUParticles* gpu_part, struct GPUEMfield* gpu_field, struct GPUGrid* gpu_grd, struct GPUParameters* gpu_param, 
                        struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param) {
    
    for (int i_sub=0; i_sub < part->n_sub_cycles; i_sub++) {

        int blockSize = THREAD_NR;
        int gridSize = ceil(part->nop / blockSize);

        particle_push_kernel<<<gridSize, blockSize>>>(gpu_part, gpu_field, gpu_grd, gpu_param);
        cudaDeviceSynchronize();
    }
}  // make different species run in parallel? subcycles themselves for the same species have to be sequential

__global__ void particle_push_kernel(struct GPUParticles* gpu_part, struct GPUEMfield* gpu_field, struct GPUGrid* gpu_grd, struct GPUParameters* gpu_param) {

    FPpart dt_sub_cycling = (FPpart) gpu_param->dt/((double) gpu_part->n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling, qomdt2 = gpu_part->qom*dto2/gpu_param->c;
    FPpart omdtsq, denom, ut, vt, wt, udotb;  // none of this should be done by each thread... constant memory?

    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;

    int ix,iy,iz;
    FPfield weight[8];
    FPfield xi[2], eta[2], zeta[2];  // lots of registers!? can you even store arrays on registers?

    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;  // stored the approximation of x^n + 1/2

    int part_ind = blockIdx.x * blockDim.x + threadIdx.x;

    xptilde = gpu_part->x[part_ind];
    yptilde = gpu_part->y[part_ind];
    zptilde = gpu_part->z[part_ind];  // initial guess for x at n + 1/2

    for(int innter=0; innter < gpu_part->NiterMover; innter++) {
        ix = 2 + int((xptilde - gpu_grd->xStart)*gpu_grd->invdx);
        iy = 2 + int((yptilde - gpu_grd->yStart)*gpu_grd->invdy);
        iz = 2 + int((zptilde - gpu_grd->zStart)*gpu_grd->invdz);  // so much indirection. is this expensive?

        int bottom_right_ind = iz * gpu_grd->nyn * gpu_grd->nxn + iy * gpu_grd->nxn + ix;  // doing a funny, might be wrong
        int top_left_ind = (iz - 1) * gpu_grd->nyn * gpu_grd->nxn + (iy - 1) * gpu_grd->nxn + (ix - 1);

        xi[0] = xptilde - gpu_grd->XN_GPU_flat[top_left_ind];
        eta[0] = yptilde - gpu_grd->YN_GPU_flat[top_left_ind];
        zeta[0] = zptilde - gpu_grd->ZN_GPU_flat[top_left_ind];
        xi[1] = xptilde - gpu_grd->XN_GPU_flat[bottom_right_ind];
        eta[1] = yptilde - gpu_grd->YN_GPU_flat[bottom_right_ind];
        zeta[1] = zptilde - gpu_grd->ZN_GPU_flat[bottom_right_ind];
        
        for (int ii = 0; ii < 2; ii++)  // another for loop... is weights[] a good idea?
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                        weight[ii * 4 + jj * 2 + kk] = xi[ii] * eta[jj] * zeta[kk] * gpu_grd->invVOL;

        Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;

        for (int ii=0; ii < 2; ii++)  // ...
            for (int jj=0; jj < 2; jj++)
                for(int kk=0; kk < 2; kk++){
                    int field_ind = (iz - kk) * gpu_grd->nyn * gpu_grd->nxn + (iy - jj) * gpu_grd->nxn + (ix - kk);  // TODO how are the flat arrays actually stored?
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

        ut = gpu_part->u[part_ind] + qomdt2 * Exl;
        vt = gpu_part->v[part_ind] + qomdt2 * Eyl;
        wt = gpu_part->w[part_ind] + qomdt2 * Ezl;
        udotb = ut * Bxl + vt * Byl + wt * Bzl;

        uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;  // velocity guess at n + 1/2
        vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
        wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;

        xptilde = gpu_part->x[part_ind] + uptilde*dto2;  // position guess at n + 1/2
        yptilde = gpu_part->y[part_ind] + vptilde*dto2;
        zptilde = gpu_part->z[part_ind] + wptilde*dto2;
    }

    gpu_part->u[part_ind] = 2.0*uptilde - gpu_part->u[part_ind];
    gpu_part->v[part_ind] = 2.0*vptilde - gpu_part->v[part_ind];
    gpu_part->w[part_ind] = 2.0*wptilde - gpu_part->w[part_ind];

    gpu_part->x[part_ind] = gpu_part->x[part_ind] + uptilde * dt_sub_cycling;
    gpu_part->y[part_ind] = gpu_part->y[part_ind] + vptilde * dt_sub_cycling;
    gpu_part->z[part_ind] = gpu_part->z[part_ind] + wptilde * dt_sub_cycling;

    if (gpu_part->x[part_ind] > gpu_grd->Lx){  // this is a disgusting amount of memory accesses!
        if (gpu_param->PERIODICX==true){ // PERIODIC
            gpu_part->x[part_ind] = gpu_part->x[part_ind] - gpu_grd->Lx;
        } else { // REFLECTING BC
            gpu_part->u[part_ind] = -gpu_part->u[part_ind];
            gpu_part->x[part_ind] = 2*gpu_grd->Lx - gpu_part->x[part_ind];
        }
    }
                                                                
    if (gpu_part->x[part_ind] < 0){
        if (gpu_param->PERIODICX==true){ // PERIODIC
            gpu_part->x[part_ind] = gpu_part->x[part_ind] + gpu_grd->Lx;
        } else { // REFLECTING BC
            gpu_part->u[part_ind] = -gpu_part->u[part_ind];
            gpu_part->x[part_ind] = -gpu_part->x[part_ind];
        }
    }
        
    
    // Y-DIRECTION: BC particles
    if (gpu_part->y[part_ind] > gpu_grd->Ly){
        if (gpu_param->PERIODICY==true){ // PERIODIC
            gpu_part->y[part_ind] = gpu_part->y[part_ind] - gpu_grd->Ly;
        } else { // REFLECTING BC
            gpu_part->v[part_ind] = -gpu_part->v[part_ind];
            gpu_part->y[part_ind] = 2*gpu_grd->Ly - gpu_part->y[part_ind];
        }
    }
                                                                
    if (gpu_part->y[part_ind] < 0){
        if (gpu_param->PERIODICY==true){ // PERIODIC
            gpu_part->y[part_ind] = gpu_part->y[part_ind] + gpu_grd->Ly;
        } else { // REFLECTING BC
            gpu_part->v[part_ind] = -gpu_part->v[part_ind];
            gpu_part->y[part_ind] = -gpu_part->y[part_ind];
        }
    }
                                                                
    // Z-DIRECTION: BC particles
    if (gpu_part->z[part_ind] > gpu_grd->Lz){
        if (gpu_param->PERIODICZ==true){ // PERIODIC
            gpu_part->z[part_ind] = gpu_part->z[part_ind] - gpu_grd->Lz;
        } else { // REFLECTING BC
            gpu_part->w[part_ind] = -gpu_part->w[part_ind];
            gpu_part->z[part_ind] = 2*gpu_grd->Lz - gpu_part->z[part_ind];
        }
    }
                                                                
    if (gpu_part->z[part_ind] < 0){
        if (gpu_param->PERIODICZ==true){ // PERIODIC
            gpu_part->z[part_ind] = gpu_part->z[part_ind] + gpu_grd->Lz;
        } else { // REFLECTING BC
            gpu_part->w[part_ind] = -gpu_part->w[part_ind];
            gpu_part->z[part_ind] = -gpu_part->z[part_ind];
        }
    }
}