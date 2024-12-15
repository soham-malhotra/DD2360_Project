#include "GPU_Interpolator.h"

#define THREAD_NR 16.0

void gpu_interpP2G(struct GPUParticles** gpu_part, struct GPUInterpDensSpecies** gpu_ids, struct GPUGrid* gpu_grd, struct particles** part, struct parameters* param) {

    for (int is=0; is < param->ns; is++){

        int blockSize = THREAD_NR;
        int gridSize = ceil((*part)[is].nop / blockSize);

        interpP2G_kernel<<<gridSize, blockSize>>>(gpu_part[is], gpu_ids[is], gpu_grd, *param);
        cudaDeviceSynchronize();
    }

}

__global__ void interpP2G_kernel(struct GPUParticles* gpu_part, struct GPUInterpDensSpecies* gpu_ids, struct GPUGrid* gpu_grd, __grid_constant__ const struct parameters param) {

    long part_ind = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (part_ind < gpu_part->nop) {

        int ix, iy, iz;
        FPpart weight;
        FPpart xi[2], eta[2], zeta[2];

        ix = 2 + int((gpu_part->x[part_ind] - gpu_grd->xStart)*gpu_grd->invdx);
        iy = 2 + int((gpu_part->y[part_ind] - gpu_grd->yStart)*gpu_grd->invdy);
        iz = 2 + int((gpu_part->z[part_ind] - gpu_grd->zStart)*gpu_grd->invdz);

        int bottom_right_ind = ix * gpu_grd->nzn * gpu_grd->nyn + iy * gpu_grd->nzn + iz;
        int top_left_ind = (ix - 1) * gpu_grd->nzn * gpu_grd->nyn + (iy - 1) * gpu_grd->nzn + (iz - 1);

        xi[0] = gpu_part->x[part_ind] - gpu_grd->XN_GPU_flat[top_left_ind];
        eta[0] = gpu_part->y[part_ind] - gpu_grd->YN_GPU_flat[top_left_ind];
        zeta[0] = gpu_part->z[part_ind] - gpu_grd->ZN_GPU_flat[top_left_ind];
        xi[1] = gpu_grd->XN_GPU_flat[bottom_right_ind] - gpu_part->x[part_ind];
        eta[1] = gpu_grd->YN_GPU_flat[bottom_right_ind] - gpu_part->y[part_ind];
        zeta[1] = gpu_grd->ZN_GPU_flat[bottom_right_ind] - gpu_part->z[part_ind];

        for (int ii=0; ii<2; ii++) {
            for (int jj=0; jj<2; jj++) {
                for (int kk=0; kk<2; kk++) {
                    weight = gpu_part->q[part_ind] * xi[ii] * eta[jj] * zeta[kk] * gpu_grd->invVOL;
                    atomicAdd(&(gpu_ids->rhon_flat[(ix - ii) * gpu_grd->nzn * gpu_grd->nyn + (iy - jj) * gpu_grd->nzn + (iz - kk)]), weight * gpu_grd->invVOL);
                    atomicAdd(&(gpu_ids->Jx_flat[(ix - ii) * gpu_grd->nzn * gpu_grd->nyn + (iy - jj) * gpu_grd->nzn + (iz - kk)]), weight * gpu_part->u[part_ind] * gpu_grd->invVOL);
                    atomicAdd(&(gpu_ids->Jy_flat[(ix - ii) * gpu_grd->nzn * gpu_grd->nyn + (iy - jj) * gpu_grd->nzn + (iz - kk)]), weight * gpu_part->v[part_ind] * gpu_grd->invVOL);
                    atomicAdd(&(gpu_ids->Jz_flat[(ix - ii) * gpu_grd->nzn * gpu_grd->nyn + (iy - jj) * gpu_grd->nzn + (iz - kk)]), weight * gpu_part->w[part_ind] * gpu_grd->invVOL);
                }
            }
        }
    }
}