#include "GPU_Interpolator.h"
#include "GPU_Particles.h"
#include <assert.h>

#define THREAD_NR 128  // multiple of 32
#define PART_SIZE_TEMP 128

void gpu_interpP2G(struct GPUParticles** gpu_part, struct GPUInterpDensSpecies** gpu_ids, struct GPUGrid* gpu_grd, struct particles** part, struct grid* grd, struct parameters* param) {
    // Create array to store streams for each species
    cudaStream_t streams[param->ns];
    
    // Create a stream for each species
    for (int is = 0; is < param->ns; is++) {
        cudaStreamCreate(&streams[is]);
    }

    // Launch kernels for each species in their respective streams
    for (int is=0; is < param->ns; is++){
        int blockSize = THREAD_NR;
        dim3 gridSize(grd->nxc - 2, grd->nyc - 2, grd->nzc - 2);  // exclude ghost cells, as there are no particles there anyways
        interpP2G_kernel<<<gridSize, blockSize, 0, streams[is]>>>(gpu_part[is], gpu_ids[is], gpu_grd);
    }

    // Wait for all streams to complete
    for (int is = 0; is < param->ns; is++) {
        cudaStreamSynchronize(streams[is]);
    }

    // Cleanup streams
    for (int is = 0; is < param->ns; is++) {
        cudaStreamDestroy(streams[is]);
    }

    // Check for any errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

__global__ void interpP2G_kernel(struct GPUParticles* gpu_part, struct GPUInterpDensSpecies* gpu_ids, struct GPUGrid* gpu_grd) {

    __shared__ FPfield XN[2], YN[2], ZN[2];  // let's have each thread individually load its particles for now
    __shared__ int cell_index, numGroups, groupPartSize;

    if (threadIdx.x == 0) {  // this is faster than anything nicer for some reason
        XN[0] = gpu_grd->XN_GPU_flat[(blockIdx.x + 1) * gpu_grd->nzn * gpu_grd->nyn + (blockIdx.y + 1) * gpu_grd->nzn + (blockIdx.z + 1)];
        YN[0] = gpu_grd->YN_GPU_flat[(blockIdx.x + 1) * gpu_grd->nzn * gpu_grd->nyn + (blockIdx.y + 1) * gpu_grd->nzn + (blockIdx.z + 1)];
        ZN[0] = gpu_grd->ZN_GPU_flat[(blockIdx.x + 1) * gpu_grd->nzn * gpu_grd->nyn + (blockIdx.y + 1) * gpu_grd->nzn + (blockIdx.z + 1)];
        XN[1] = gpu_grd->XN_GPU_flat[(blockIdx.x + 2) * gpu_grd->nzn * gpu_grd->nyn + (blockIdx.y + 2) * gpu_grd->nzn + (blockIdx.z + 2)];
        YN[1] = gpu_grd->YN_GPU_flat[(blockIdx.x + 2) * gpu_grd->nzn * gpu_grd->nyn + (blockIdx.y + 2) * gpu_grd->nzn + (blockIdx.z + 2)];
        ZN[1] = gpu_grd->ZN_GPU_flat[(blockIdx.x + 2) * gpu_grd->nzn * gpu_grd->nyn + (blockIdx.y + 2) * gpu_grd->nzn + (blockIdx.z + 2)];

        cell_index = (blockIdx.x + 1) * gpu_grd->nyc * gpu_grd->nzc + (blockIdx.y + 1) * gpu_grd->nzc + (blockIdx.z + 1);
        numGroups = blockDim.x / 8;
        groupPartSize = gpu_part->cell_counter[cell_index] / numGroups;
    }

    __syncthreads();  // TODO not sure if necessary?

    FPfield accumulateRho = 0;
    FPfield accumulateJx = 0;
    FPfield accumulateJy = 0;
    FPfield accumulateJz = 0;

    int groupId = threadIdx.x / 8;

    int start_index = cell_index * MAX_PART_PER_CELL + groupId * groupPartSize;
    int end_index = (groupId == numGroups - 1) ? cell_index * MAX_PART_PER_CELL + gpu_part->cell_counter[cell_index] : start_index + groupPartSize;
    
    int nodeId = threadIdx.x & 7;  // faster than % 8
    int i = nodeId >> 2;           // faster than / 4
    int j = (nodeId >> 1) & 1;     // faster than (nodeId % 4) / 2
    int k = nodeId & 1;            // faster than % 2
    
    FPfield temp_inc = gpu_grd -> invVOL * gpu_grd -> invVOL * ((i + j + k == 1 || i + j + k == 3) ? -1.0 : 1.0);

    for(long part_ind = start_index; part_ind < end_index; part_ind++) {

        FPfield temp = gpu_part->cell_q[part_ind] * temp_inc * (gpu_part->cell_x[part_ind] - XN[i]) * (gpu_part->cell_y[part_ind] - YN[j]) * (gpu_part->cell_z[part_ind] - ZN[k]);

        accumulateRho += temp;
        accumulateJx += temp * gpu_part->cell_u[part_ind];
        accumulateJy += temp * gpu_part->cell_v[part_ind];
        accumulateJz += temp * gpu_part->cell_w[part_ind];
    }

    int flatIndex = ((blockIdx.x + 2) - i) * gpu_grd->nzn * gpu_grd->nyn + ((blockIdx.y + 2) - j) * gpu_grd->nzn + ((blockIdx.z + 2) - k);
    atomicAdd(&(gpu_ids->rhon_flat[flatIndex]), accumulateRho);
    atomicAdd(&(gpu_ids->Jx_flat[flatIndex]), accumulateJx);
    atomicAdd(&(gpu_ids->Jy_flat[flatIndex]), accumulateJy);
    atomicAdd(&(gpu_ids->Jz_flat[flatIndex]), accumulateJz);
}
