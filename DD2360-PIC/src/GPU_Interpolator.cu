#include "GPU_Interpolator.h"
#include <assert.h>

#define THREAD_NR 128  // multiple of 32
#define PART_SIZE_TEMP 128

void gpu_interpP2G(struct GPUParticles** gpu_part, struct GPUInterpDensSpecies** gpu_ids, struct GPUGrid* gpu_grd, struct particles** part, struct grid* grd, struct parameters* param) {

    for (int is=0; is < param->ns; is++){

        int blockSize = THREAD_NR;
        dim3 gridSize(grd->nxc - 2, grd->nyc - 2, grd->nzc - 2);  // exclude ghost cells, as there are no particles there anyways (?)
        // int gridSize = ceil((*part)[is].nop / blockSize);

        interpP2G_kernel<<<gridSize, blockSize>>>(gpu_part[is], gpu_ids[is], gpu_grd);
        cudaDeviceSynchronize();
    }
}

// __global__ void interpP2G_kernel(struct GPUParticles* gpu_part, struct GPUInterpDensSpecies* gpu_ids, struct GPUGrid* gpu_grd) {  // TODO constant memory!

//     __shared__ FPfield cell_rho[8], cell_Jx[8], cell_Jy[8], cell_Jz[8];
//     __shared__ FPfield XN[2], YN[2], ZN[2];
//     __shared__ FPpart x[THREAD_NR], y[THREAD_NR], z[THREAD_NR], u[THREAD_NR], v[THREAD_NR], w[THREAD_NR];
//     __shared__ FPfield xi[THREAD_NR][2], eta[THREAD_NR][2], zeta[THREAD_NR][2];

//     if (threadIdx.x < 8) {
//         cell_rho[threadIdx.x] = 0;
//         cell_Jx[threadIdx.x] = 0;
//         cell_Jy[threadIdx.x] = 0;
//         cell_Jz[threadIdx.x] = 0;
//     }

//     __syncthreads();

//     if (threadIdx.x < 6) {  // first 6 threads load cell coordinates
//         FPfield* N;
//         FPfield* N_GPU_GLAT;
    
//         if (threadIdx.x / 2 == 0) {  // these are just pointers!
//             N = XN;
//             N_GPU_GLAT = gpu_grd->XN_GPU_flat;
//         } else if (threadIdx.x / 2 == 1) {
//             N = YN;
//             N_GPU_GLAT = gpu_grd->YN_GPU_flat;
//         } else {
//             N = ZN;
//             N_GPU_GLAT = gpu_grd->ZN_GPU_flat;
//         }
        
//         N[threadIdx.x % 2] = N_GPU_GLAT[(blockIdx.x + threadIdx.x % 2 + 1) * gpu_grd->nzn * gpu_grd->nyn + 
//                         (blockIdx.y + threadIdx.x % 2 + 1) * gpu_grd->nzn + 
//                         (blockIdx.z + threadIdx.x % 2 + 1)];
//     }

//     __syncthreads();

//     for(long part_ind = threadIdx.x; part_ind < PART_SIZE_TEMP; part_ind += blockDim.x) {
//           // TODO this is a disaster, use some sort of data structure
//         if (true) {  //(gpu_part->cell_id[part_ind] == (blockIdx.x + 1) * gpu_grd->nzc * gpu_grd->nyc + (blockIdx.y + 1) * gpu_grd->nzc + (blockIdx.z + 1)) {

//             FPpart weight;
            
//             x[threadIdx.x] = gpu_part->x[part_ind];
//             y[threadIdx.x] = gpu_part->y[part_ind];
//             z[threadIdx.x] = gpu_part->z[part_ind];
//             u[threadIdx.x] = gpu_part->u[part_ind];
//             v[threadIdx.x] = gpu_part->v[part_ind];
//             w[threadIdx.x] = gpu_part->w[part_ind];

//             xi[threadIdx.x][0] = x[threadIdx.x] - XN[0];
//             eta[threadIdx.x][0] = y[threadIdx.x] - YN[0];
//             zeta[threadIdx.x][0] = z[threadIdx.x] - ZN[0];
//             xi[threadIdx.x][1] = XN[1] - x[threadIdx.x];
//             eta[threadIdx.x][1] = YN[1] - y[threadIdx.x];
//             zeta[threadIdx.x][1] = ZN[1] - z[threadIdx.x];

//             weight = gpu_part->q[part_ind] * gpu_grd -> invVOL * gpu_grd -> invVOL;  // saved into register to avoid repeated access

//             for(int i = 0; i < 2; i++)
//                 for(int j = 0; j < 2; j++)
//                     for(int k = 0; k < 2; k++) {
//                         atomicAdd(&(cell_rho[i * 4 + j * 2 + k]), weight * xi[threadIdx.x][i] * eta[threadIdx.x][j] * zeta[threadIdx.x][k]);
//                         atomicAdd(&(cell_Jx[i * 4 + j * 2 + k]), weight * xi[threadIdx.x][i] * eta[threadIdx.x][j] * zeta[threadIdx.x][k] * u[threadIdx.x]);
//                         atomicAdd(&(cell_Jy[i * 4 + j * 2 + k]), weight * xi[threadIdx.x][i] * eta[threadIdx.x][j] * zeta[threadIdx.x][k] * v[threadIdx.x]);
//                         atomicAdd(&(cell_Jz[i * 4 + j * 2 + k]), weight * xi[threadIdx.x][i] * eta[threadIdx.x][j] * zeta[threadIdx.x][k] * w[threadIdx.x]);
//                     }
//         }
//     }

//     __syncthreads();

//     if(threadIdx.x < 8) {  // fist 8 threads transfer shared memory to global
//         atomicAdd(&(gpu_ids->rhon_flat[((blockIdx.x + 2) - threadIdx.x / 4) * gpu_grd->nzn * gpu_grd->nyn + ((blockIdx.y + 2) - (threadIdx.x % 4) / 2) * gpu_grd->nzn + ((blockIdx.z + 2) - threadIdx.x % 2)]), cell_rho[threadIdx.x]);
//         atomicAdd(&(gpu_ids->Jx_flat[((blockIdx.x + 2) - threadIdx.x / 4) * gpu_grd->nzn * gpu_grd->nyn + ((blockIdx.y + 2) - (threadIdx.x % 4) / 2) * gpu_grd->nzn + ((blockIdx.z + 2) - threadIdx.x % 2)]), cell_Jx[threadIdx.x]);
//         atomicAdd(&(gpu_ids->Jy_flat[((blockIdx.x + 2) - threadIdx.x / 4) * gpu_grd->nzn * gpu_grd->nyn + ((blockIdx.y + 2) - (threadIdx.x % 4) / 2) * gpu_grd->nzn + ((blockIdx.z + 2) - threadIdx.x % 2)]), cell_Jy[threadIdx.x]);
//         atomicAdd(&(gpu_ids->Jz_flat[((blockIdx.x + 2) - threadIdx.x / 4) * gpu_grd->nzn * gpu_grd->nyn + ((blockIdx.y + 2) - (threadIdx.x % 4) / 2) * gpu_grd->nzn + ((blockIdx.z + 2) - threadIdx.x % 2)]), cell_Jz[threadIdx.x]);
//     }
// }

__global__ void interpP2G_kernel(struct GPUParticles* gpu_part, struct GPUInterpDensSpecies* gpu_ids, struct GPUGrid* gpu_grd) {

    __shared__ FPfield XN[2], YN[2], ZN[2];  // let's have each thread individually load its particles for now

    if (threadIdx.x < 6) {  // first 6 threads load cell coordinates
        FPfield* N;
        FPfield* N_GPU_GLAT;
    
        if (threadIdx.x / 2 == 0) {  // these are just pointers!
            N = XN;
            N_GPU_GLAT = gpu_grd->XN_GPU_flat;
        } else if (threadIdx.x / 2 == 1) {
            N = YN;
            N_GPU_GLAT = gpu_grd->YN_GPU_flat;
        } else {
            N = ZN;
            N_GPU_GLAT = gpu_grd->ZN_GPU_flat;
        }
        
        N[threadIdx.x % 2] = N_GPU_GLAT[(blockIdx.x + threadIdx.x % 2 + 1) * gpu_grd->nzn * gpu_grd->nyn + 
                        (blockIdx.y + threadIdx.x % 2 + 1) * gpu_grd->nzn + 
                        (blockIdx.z + threadIdx.x % 2 + 1)];
    }

    FPfield accumulateRho = 0;
    FPfield accumulateJx = 0;
    FPfield accumulateJy = 0;
    FPfield accumulateJz = 0;

    int warpId = threadIdx.x / 32;
    int numWarps = blockDim.x / 32;
    int quantId = (threadIdx.x % 32) / 8;
    int cellId = threadIdx.x % 8;

    printf("Thread %d, warp %d, quant %d, cell %d\n", threadIdx.x, warpId, quantId, cellId);

    int i = cellId / 4;  // 0 or 1
    int j = (cellId % 4) / 2;   // 0 or 1
    int k = cellId % 2;  // 0 or 1

    int warpPartSize = PART_SIZE_TEMP / numWarps;

    for(long part_ind = warpId * warpPartSize; part_ind < (warpId + 1) * warpPartSize; part_ind++) {
        // TODO iterate from first to last particle in current block

        FPfield vels[4]; // shared?
        vels[0] = 1;
        vels[1] = gpu_part->u[part_ind];
        vels[2] = gpu_part->v[part_ind];
        vels[3] = gpu_part->w[part_ind];

        accumulate += gpu_part->q[part_ind] * gpu_grd -> invVOL * gpu_grd -> invVOL * vels[quantId] * (gpu_part->x[part_ind] - XN[i]) * (gpu_part->y[part_ind] - YN[j]) * (gpu_part->z[part_ind] - ZN[k]);  // TODO signs!

    }

    FPfield* quants[4];  // shared?
    quants[0] = &(gpu_ids->rhon_flat[((blockIdx.x + 2) - i) * gpu_grd->nzn * gpu_grd->nyn + ((blockIdx.y + 2) - j) * gpu_grd->nzn + ((blockIdx.z + 2) - k)]);
    quants[1] = &(gpu_ids->Jx_flat[((blockIdx.x + 2) - i) * gpu_grd->nzn * gpu_grd->nyn + ((blockIdx.y + 2) - j) * gpu_grd->nzn + ((blockIdx.z + 2) - k)]);
    quants[2] = &(gpu_ids->Jy_flat[((blockIdx.x + 2) - i) * gpu_grd->nzn * gpu_grd->nyn + ((blockIdx.y + 2) - j) * gpu_grd->nzn + ((blockIdx.z + 2) - k)]);
    quants[3] = &(gpu_ids->Jz_flat[((blockIdx.x + 2) - i) * gpu_grd->nzn * gpu_grd->nyn + ((blockIdx.y + 2) - j) * gpu_grd->nzn + ((blockIdx.z + 2) - k)]);

    atomicAdd(quants[quantId], accumulate);
}
