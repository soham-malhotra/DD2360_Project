#include "GPU_Interpolator.h"
#include "GPU_Particles.h"
#include <assert.h>

#define THREAD_NR 128  // multiple of 32
#define PART_SIZE_TEMP 128

void gpu_interpP2G(struct GPUParticles** gpu_part, struct GPUInterpDensSpecies** gpu_ids, struct GPUGrid* gpu_grd, struct particles** part, struct grid* grd, struct parameters* param) {

    for (int is=0; is < param->ns; is++){

        int blockSize = THREAD_NR;
        dim3 gridSize(grd->nxc - 2, grd->nyc - 2, grd->nzc - 2);  // exclude ghost cells, as there are no particles there anyways

        interpP2G_kernel<<<gridSize, blockSize>>>(gpu_part[is], gpu_ids[is], gpu_grd);
        cudaDeviceSynchronize();
    }
}

__global__ void interpP2G_kernel(struct GPUParticles* gpu_part, struct GPUInterpDensSpecies* gpu_ids, struct GPUGrid* gpu_grd) {

    __shared__ FPfield XN[2], YN[2], ZN[2];  // let's have each thread individually load its particles for now

    if (threadIdx.x == 0) {  // this is faster than anything nicer for some reason
        XN[0] = gpu_grd->XN_GPU_flat[(blockIdx.x + 1) * gpu_grd->nzn * gpu_grd->nyn + (blockIdx.y + 1) * gpu_grd->nzn + (blockIdx.z + 1)];
        YN[0] = gpu_grd->YN_GPU_flat[(blockIdx.x + 1) * gpu_grd->nzn * gpu_grd->nyn + (blockIdx.y + 1) * gpu_grd->nzn + (blockIdx.z + 1)];
        ZN[0] = gpu_grd->ZN_GPU_flat[(blockIdx.x + 1) * gpu_grd->nzn * gpu_grd->nyn + (blockIdx.y + 1) * gpu_grd->nzn + (blockIdx.z + 1)];
        XN[1] = gpu_grd->XN_GPU_flat[(blockIdx.x + 2) * gpu_grd->nzn * gpu_grd->nyn + (blockIdx.y + 2) * gpu_grd->nzn + (blockIdx.z + 2)];
        YN[1] = gpu_grd->YN_GPU_flat[(blockIdx.x + 2) * gpu_grd->nzn * gpu_grd->nyn + (blockIdx.y + 2) * gpu_grd->nzn + (blockIdx.z + 2)];
        ZN[1] = gpu_grd->ZN_GPU_flat[(blockIdx.x + 2) * gpu_grd->nzn * gpu_grd->nyn + (blockIdx.y + 2) * gpu_grd->nzn + (blockIdx.z + 2)];
    }

    FPfield accumulateRho = 0;
    FPfield accumulateJx = 0;
    FPfield accumulateJy = 0;
    FPfield accumulateJz = 0;

    int cell_index = (blockIdx.x + 1) * gpu_grd->nyc * gpu_grd->nzc + (blockIdx.y + 1) * gpu_grd->nzc + (blockIdx.z + 1);
    int start_index = cell_index * MAX_PART_PER_CELL;

    int groupId = threadIdx.x / 8;
    int groupPartSize = gpu_part->cell_counter[cell_index] / (blockDim.x / 8);
    int cellId = threadIdx.x % 8;
    int i = cellId / 4;  // 0 or 1
    int j = (cellId % 4) / 2;   // 0 or 1
    int k = cellId % 2;  // 0 or 1
    FPfield temp_inc = gpu_grd -> invVOL * gpu_grd -> invVOL * ((i + j + k == 1 || i + j + k == 3) ? -1.0 : 1.0);

    for(long part_ind = start_index + groupId * groupPartSize; part_ind < start_index + (groupId + 1) * groupPartSize; part_ind++) {  // TODO for 32 threads, 11 particles may be split into 2 groups of 4 -> the last 3 is not processed!
        FPfield temp = gpu_part->q[part_ind] * temp_inc * (gpu_part->x[part_ind] - XN[i]) * (gpu_part->y[part_ind] - YN[j]) * (gpu_part->z[part_ind] - ZN[k]);

        accumulateRho += temp;
        accumulateJx += temp * gpu_part->u[part_ind];
        accumulateJy += temp * gpu_part->v[part_ind];
        accumulateJz += temp * gpu_part->w[part_ind];
    }

    int flatIndex = ((blockIdx.x + 2) - i) * gpu_grd->nzn * gpu_grd->nyn + ((blockIdx.y + 2) - j) * gpu_grd->nzn + ((blockIdx.z + 2) - k);
    atomicAdd(&(gpu_ids->rhon_flat[flatIndex]), accumulateRho);
    atomicAdd(&(gpu_ids->Jx_flat[flatIndex]), accumulateJx);
    atomicAdd(&(gpu_ids->Jy_flat[flatIndex]), accumulateJy);
    atomicAdd(&(gpu_ids->Jz_flat[flatIndex]), accumulateJz);
}
