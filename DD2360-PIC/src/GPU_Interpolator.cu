#include "GPU_Interpolator.h"

#define THREAD_NR 256.0

void gpu_interpP2G(struct GPUParticles** gpu_part, struct GPUInterpDensSpecies** gpu_ids, struct GPUGrid* gpu_grd, struct particles** part, struct grid* grd, struct parameters* param) {

    for (int is=0; is < param->ns; is++){

        int blockSize = THREAD_NR;
        dim3 gridSize(grd->nxn - 2, grd->nyn - 2, grd->nzn - 2);  // exclude ghost cells, as there are no particles there anyways (?)
        // int gridSize = ceil((*part)[is].nop / blockSize);

        interpP2G_kernel<<<gridSize, blockSize>>>(gpu_part[is], gpu_ids[is], gpu_grd);
        cudaDeviceSynchronize();
    }
}

// __global__ void interpP2G_kernel(struct GPUParticles* gpu_part, struct GPUInterpDensSpecies* gpu_ids, struct GPUGrid* gpu_grd) {

//     long part_ind = blockIdx.x * blockDim.x + threadIdx.x;
    
//     if (part_ind < gpu_part->nop) {

//         int ix, iy, iz;
//         FPpart weight;
//         FPpart xi[2], eta[2], zeta[2];

//         FPpart x, y, z, u, v, w;
//         x = gpu_part->x[part_ind];
//         y = gpu_part->y[part_ind];
//         z = gpu_part->z[part_ind];
//         u = gpu_part->u[part_ind];
//         v = gpu_part->v[part_ind];
//         w = gpu_part->w[part_ind];

//         ix = 2 + int((x - gpu_grd->xStart)*gpu_grd->invdx); // TODO store in particles!
//         iy = 2 + int((y - gpu_grd->yStart)*gpu_grd->invdy);
//         iz = 2 + int((z - gpu_grd->zStart)*gpu_grd->invdz);

//         xi[0] = x - gpu_grd->XN_GPU_flat[(ix - 1) * gpu_grd->nzn * gpu_grd->nyn + (iy - 1) * gpu_grd->nzn + (iz - 1)];
//         eta[0] = y - gpu_grd->YN_GPU_flat[(ix - 1) * gpu_grd->nzn * gpu_grd->nyn + (iy - 1) * gpu_grd->nzn + (iz - 1)];
//         zeta[0] = z - gpu_grd->ZN_GPU_flat[(ix - 1) * gpu_grd->nzn * gpu_grd->nyn + (iy - 1) * gpu_grd->nzn + (iz - 1)];
//         xi[1] = gpu_grd->XN_GPU_flat[ix * gpu_grd->nzn * gpu_grd->nyn + iy * gpu_grd->nzn + iz] - x;
//         eta[1] = gpu_grd->YN_GPU_flat[ix * gpu_grd->nzn * gpu_grd->nyn + iy * gpu_grd->nzn + iz] - y;
//         zeta[1] = gpu_grd->ZN_GPU_flat[ix * gpu_grd->nzn * gpu_grd->nyn + iy * gpu_grd->nzn + iz] - z;

//         weight = gpu_part->q[part_ind] * gpu_grd -> invVOL * gpu_grd -> invVOL;  // saved into register to avoid repeated access

//         // ii = 0, jj = 0, kk = 0
//         atomicAdd(&(gpu_ids->rhon_flat[(ix - 0) * gpu_grd->nzn * gpu_grd->nyn + (iy - 0) * gpu_grd->nzn + (iz - 0)]), weight * xi[0] * eta[0] * zeta[0]);
//         atomicAdd(&(gpu_ids->Jx_flat[(ix - 0) * gpu_grd->nzn * gpu_grd->nyn + (iy - 0) * gpu_grd->nzn + (iz - 0)]), weight * xi[0] * eta[0] * zeta[0] * u);
//         atomicAdd(&(gpu_ids->Jy_flat[(ix - 0) * gpu_grd->nzn * gpu_grd->nyn + (iy - 0) * gpu_grd->nzn + (iz - 0)]), weight * xi[0] * eta[0] * zeta[0] * v);
//         atomicAdd(&(gpu_ids->Jz_flat[(ix - 0) * gpu_grd->nzn * gpu_grd->nyn + (iy - 0) * gpu_grd->nzn + (iz - 0)]), weight * xi[0] * eta[0] * zeta[0] * w);

//         // ii = 0, jj = 0, kk = 1
//         atomicAdd(&(gpu_ids->rhon_flat[(ix - 0) * gpu_grd->nzn * gpu_grd->nyn + (iy - 0) * gpu_grd->nzn + (iz - 1)]), weight * xi[0] * eta[0] * zeta[1]);
//         atomicAdd(&(gpu_ids->Jx_flat[(ix - 0) * gpu_grd->nzn * gpu_grd->nyn + (iy - 0) * gpu_grd->nzn + (iz - 1)]), weight * xi[0] * eta[0] * zeta[1] * u);
//         atomicAdd(&(gpu_ids->Jy_flat[(ix - 0) * gpu_grd->nzn * gpu_grd->nyn + (iy - 0) * gpu_grd->nzn + (iz - 1)]), weight * xi[0] * eta[0] * zeta[1] * v);
//         atomicAdd(&(gpu_ids->Jz_flat[(ix - 0) * gpu_grd->nzn * gpu_grd->nyn + (iy - 0) * gpu_grd->nzn + (iz - 1)]), weight * xi[0] * eta[0] * zeta[1] * w);

//         // ii = 0, jj = 1, kk = 0
//         atomicAdd(&(gpu_ids->rhon_flat[(ix - 0) * gpu_grd->nzn * gpu_grd->nyn + (iy - 1) * gpu_grd->nzn + (iz - 0)]), weight * xi[0] * eta[1] * zeta[0]);
//         atomicAdd(&(gpu_ids->Jx_flat[(ix - 0) * gpu_grd->nzn * gpu_grd->nyn + (iy - 1) * gpu_grd->nzn + (iz - 0)]), weight * xi[0] * eta[1] * zeta[0] * u);
//         atomicAdd(&(gpu_ids->Jy_flat[(ix - 0) * gpu_grd->nzn * gpu_grd->nyn + (iy - 1) * gpu_grd->nzn + (iz - 0)]), weight * xi[0] * eta[1] * zeta[0] * v);
//         atomicAdd(&(gpu_ids->Jz_flat[(ix - 0) * gpu_grd->nzn * gpu_grd->nyn + (iy - 1) * gpu_grd->nzn + (iz - 0)]), weight * xi[0] * eta[1] * zeta[0] * w);

//         // ii = 0, jj = 1, kk = 1
//         atomicAdd(&(gpu_ids->rhon_flat[(ix - 0) * gpu_grd->nzn * gpu_grd->nyn + (iy - 1) * gpu_grd->nzn + (iz - 1)]), weight * xi[0] * eta[1] * zeta[1]);
//         atomicAdd(&(gpu_ids->Jx_flat[(ix - 0) * gpu_grd->nzn * gpu_grd->nyn + (iy - 1) * gpu_grd->nzn + (iz - 1)]), weight * xi[0] * eta[1] * zeta[1] * u);
//         atomicAdd(&(gpu_ids->Jy_flat[(ix - 0) * gpu_grd->nzn * gpu_grd->nyn + (iy - 1) * gpu_grd->nzn + (iz - 1)]), weight * xi[0] * eta[1] * zeta[1] * v);
//         atomicAdd(&(gpu_ids->Jz_flat[(ix - 0) * gpu_grd->nzn * gpu_grd->nyn + (iy - 1) * gpu_grd->nzn + (iz - 1)]), weight * xi[0] * eta[1] * zeta[1] * w);

//         // ii = 1, jj = 0, kk = 0
//         atomicAdd(&(gpu_ids->rhon_flat[(ix - 1) * gpu_grd->nzn * gpu_grd->nyn + (iy - 0) * gpu_grd->nzn + (iz - 0)]), weight * xi[1] * eta[0] * zeta[0]);
//         atomicAdd(&(gpu_ids->Jx_flat[(ix - 1) * gpu_grd->nzn * gpu_grd->nyn + (iy - 0) * gpu_grd->nzn + (iz - 0)]), weight * xi[1] * eta[0] * zeta[0] * u);
//         atomicAdd(&(gpu_ids->Jy_flat[(ix - 1) * gpu_grd->nzn * gpu_grd->nyn + (iy - 0) * gpu_grd->nzn + (iz - 0)]), weight * xi[1] * eta[0] * zeta[0] * v);
//         atomicAdd(&(gpu_ids->Jz_flat[(ix - 1) * gpu_grd->nzn * gpu_grd->nyn + (iy - 0) * gpu_grd->nzn + (iz - 0)]), weight * xi[1] * eta[0] * zeta[0] * w);

//         // ii = 1, jj = 0, kk = 1
//         atomicAdd(&(gpu_ids->rhon_flat[(ix - 1) * gpu_grd->nzn * gpu_grd->nyn + (iy - 0) * gpu_grd->nzn + (iz - 1)]), weight * xi[1] * eta[0] * zeta[1]);
//         atomicAdd(&(gpu_ids->Jx_flat[(ix - 1) * gpu_grd->nzn * gpu_grd->nyn + (iy - 0) * gpu_grd->nzn + (iz - 1)]), weight * xi[1] * eta[0] * zeta[1] * u);
//         atomicAdd(&(gpu_ids->Jy_flat[(ix - 1) * gpu_grd->nzn * gpu_grd->nyn + (iy - 0) * gpu_grd->nzn + (iz - 1)]), weight * xi[1] * eta[0] * zeta[1] * v);
//         atomicAdd(&(gpu_ids->Jz_flat[(ix - 1) * gpu_grd->nzn * gpu_grd->nyn + (iy - 0) * gpu_grd->nzn + (iz - 1)]), weight * xi[1] * eta[0] * zeta[1] * w);

//         // ii = 1, jj = 1, kk = 0
//         atomicAdd(&(gpu_ids->rhon_flat[(ix - 1) * gpu_grd->nzn * gpu_grd->nyn + (iy - 1) * gpu_grd->nzn + (iz - 0)]),  weight * xi[1] * eta[1] * zeta[0]);
//         atomicAdd(&(gpu_ids->Jx_flat[(ix - 1) * gpu_grd->nzn * gpu_grd->nyn + (iy - 1) * gpu_grd->nzn + (iz - 0)]),  weight * xi[1] * eta[1] * zeta[0] * u);
//         atomicAdd(&(gpu_ids->Jy_flat[(ix - 1) * gpu_grd->nzn * gpu_grd->nyn + (iy - 1) * gpu_grd->nzn + (iz - 0)]),  weight * xi[1] * eta[1] * zeta[0] * v);
//         atomicAdd(&(gpu_ids->Jz_flat[(ix - 1) * gpu_grd->nzn * gpu_grd->nyn + (iy - 1) * gpu_grd->nzn + (iz - 0)]),  weight * xi[1] * eta[1] * zeta[0] * w);

//         // ii = 1, jj = 1, kk = 1
//         atomicAdd(&(gpu_ids->rhon_flat[(ix - 1) * gpu_grd->nzn * gpu_grd->nyn + (iy - 1) * gpu_grd->nzn + (iz - 1)]), weight * xi[1] * eta[1] * zeta[1]);
//         atomicAdd(&(gpu_ids->Jx_flat[(ix - 1) * gpu_grd->nzn * gpu_grd->nyn + (iy - 1) * gpu_grd->nzn + (iz - 1)]), weight * xi[1] * eta[1] * zeta[1] * u);
//         atomicAdd(&(gpu_ids->Jy_flat[(ix - 1) * gpu_grd->nzn * gpu_grd->nyn + (iy - 1) * gpu_grd->nzn + (iz - 1)]), weight * xi[1] * eta[1] * zeta[1] * v);
//         atomicAdd(&(gpu_ids->Jz_flat[(ix - 1) * gpu_grd->nzn * gpu_grd->nyn + (iy - 1) * gpu_grd->nzn + (iz - 1)]), weight * xi[1] * eta[1] * zeta[1] * w);
//     }
// }

__global__ void interpP2G_kernel(struct GPUParticles* gpu_part, struct GPUInterpDensSpecies* gpu_ids, struct GPUGrid* gpu_grd) {  // TODO constant memory!

    FPpart weight;
    FPpart xi[2], eta[2], zeta[2];
    FPpart x, y, z, u, v, w;

    __shared__ FPfield cell_rho[8];
    __shared__ FPfield cell_Jx[8];
    __shared__ FPfield cell_Jy[8];
    __shared__ FPfield cell_Jz[8];

    if (threadIdx.x < 8) {  // first 8 threads zero shared memory
        cell_rho[threadIdx.x] = 0.0;
        cell_Jx[threadIdx.x] = 0.0;
        cell_Jy[threadIdx.x] = 0.0;
        cell_Jz[threadIdx.x] = 0.0;
    }

    __syncthreads();

    for(long part_ind = threadIdx.x; part_ind < gpu_part->nop; part_ind += blockDim.x) {  // TODO this is a disaster, use some sort of data structure
        if (gpu_part->cell_id[part_ind] == (blockIdx.x + 1) * blockDim.y * blockDim.z + (blockIdx.x + 1) * blockDim.z + (blockIdx.z + 1)) {
            
            x = gpu_part->x[part_ind];
            y = gpu_part->y[part_ind];
            z = gpu_part->z[part_ind];
            u = gpu_part->u[part_ind];
            v = gpu_part->v[part_ind];
            w = gpu_part->w[part_ind];

            xi[0] = x - gpu_grd->XN_GPU_flat[((blockIdx.x + 1) - 1) * gpu_grd->nzn * gpu_grd->nyn + ((blockIdx.x + 1) - 1) * gpu_grd->nzn + ((blockIdx.z + 1) - 1)];  // TODO redundant accesses
            eta[0] = y - gpu_grd->YN_GPU_flat[((blockIdx.x + 1) - 1) * gpu_grd->nzn * gpu_grd->nyn + ((blockIdx.x + 1) - 1) * gpu_grd->nzn + ((blockIdx.z + 1) - 1)];
            zeta[0] = z - gpu_grd->ZN_GPU_flat[((blockIdx.x + 1) - 1) * gpu_grd->nzn * gpu_grd->nyn + ((blockIdx.x + 1) - 1) * gpu_grd->nzn + ((blockIdx.z + 1) - 1)];
            xi[1] = gpu_grd->XN_GPU_flat[(blockIdx.x + 1) * gpu_grd->nzn * gpu_grd->nyn + (blockIdx.x + 1) * gpu_grd->nzn + (blockIdx.z + 1)] - x;
            eta[1] = gpu_grd->YN_GPU_flat[(blockIdx.x + 1) * gpu_grd->nzn * gpu_grd->nyn + (blockIdx.x + 1) * gpu_grd->nzn + (blockIdx.z + 1)] - y;
            zeta[1] = gpu_grd->ZN_GPU_flat[(blockIdx.x + 1) * gpu_grd->nzn * gpu_grd->nyn + (blockIdx.x + 1) * gpu_grd->nzn + (blockIdx.z + 1)] - z;

            weight = gpu_part->q[part_ind] * gpu_grd -> invVOL * gpu_grd -> invVOL;  // saved into register to avoid repeated access

            for(int i = 0; i < 2; i++)
                for(int j = 0; j < 2; j++)
                    for(int k = 0; k < 2; k++) {
                        atomicAdd(&(cell_rho[i * 4 + j * 2 + k]), weight * xi[i] * eta[j] * zeta[k]);
                        atomicAdd(&(cell_Jx[i * 4 + j * 2 + k]), weight * xi[i] * eta[j] * zeta[k] * u);
                        atomicAdd(&(cell_Jy[i * 4 + j * 2 + k]), weight * xi[i] * eta[j] * zeta[k] * v);
                        atomicAdd(&(cell_Jz[i * 4 + j * 2 + k]), weight * xi[i] * eta[j] * zeta[k] * w);
                    }
        }
    }

    __syncthreads();

    if(threadIdx.x < 8) {  // fist 8 threads transfer shared memory to global
        int i = threadIdx.x / 4;
        int j = (threadIdx.x % 4) / 2;
        int k = threadIdx.x % 2;

        atomicAdd(&(gpu_ids->rhon_flat[((blockIdx.x + 1) - i) * gpu_grd->nzn * gpu_grd->nyn + ((blockIdx.x + 1) - j) * gpu_grd->nzn + ((blockIdx.z + 1) - k)]), cell_rho[threadIdx.x]);
        atomicAdd(&(gpu_ids->Jx_flat[((blockIdx.x + 1) - i) * gpu_grd->nzn * gpu_grd->nyn + ((blockIdx.x + 1) - j) * gpu_grd->nzn + ((blockIdx.z + 1) - k)]), cell_Jx[threadIdx.x]);
        atomicAdd(&(gpu_ids->Jy_flat[((blockIdx.x + 1) - i) * gpu_grd->nzn * gpu_grd->nyn + ((blockIdx.x + 1) - j) * gpu_grd->nzn + ((blockIdx.z + 1) - k)]), cell_Jy[threadIdx.x]);
        atomicAdd(&(gpu_ids->Jz_flat[((blockIdx.x + 1) - i) * gpu_grd->nzn * gpu_grd->nyn + ((blockIdx.x + 1) - j) * gpu_grd->nzn + ((blockIdx.z + 1) - k)]), cell_Jz[threadIdx.x]);
    }
}