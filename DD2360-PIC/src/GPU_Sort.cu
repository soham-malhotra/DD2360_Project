#include "GPU_Sort.h"

#define THREAD_NR 128.0

// implement a kernel for sorting the particle arrays here!

void gpu_sort_particles(struct GPUParticles** gpu_part, struct particles* part, struct parameters* param, struct grid* grd) {

    for (int is=0; is < param->ns; is++) {
        
        int blockSize = THREAD_NR;
        int gridSize = ceil(part[is].nop / blockSize);

        // reset cell counter
        int* cell_counter_ptr;  // copy pointer to host
        cudaErrorHandling(cudaMemcpy(&cell_counter_ptr, &(gpu_part[is]->cell_counter), sizeof(int*), cudaMemcpyDeviceToHost));  // TODO cringe
        cudaErrorHandling(cudaMemset(cell_counter_ptr, 0, grd->nxc * grd->nyc * grd->nzc * sizeof(int)));

        categorize_kernel<<<gridSize, blockSize>>>(*gpu_part);
    }
}


__global__ void categorize_kernel(struct GPUParticles* gpu_part) {

    int part_ind = blockIdx.x * blockDim.x + threadIdx.x;
    int cell_id = gpu_part->cell_id[part_ind];

    int cell_pos = atomicAdd(&gpu_part->cell_counter[cell_id], 1);
    gpu_part->cell_x[cell_id * MAX_PART_PER_CELL + cell_pos] = gpu_part->x[part_ind];
    gpu_part->cell_y[cell_id * MAX_PART_PER_CELL + cell_pos] = gpu_part->y[part_ind];
    gpu_part->cell_z[cell_id * MAX_PART_PER_CELL + cell_pos] = gpu_part->z[part_ind];
    gpu_part->cell_u[cell_id * MAX_PART_PER_CELL + cell_pos] = gpu_part->u[part_ind];
    gpu_part->cell_v[cell_id * MAX_PART_PER_CELL + cell_pos] = gpu_part->v[part_ind];
    gpu_part->cell_w[cell_id * MAX_PART_PER_CELL + cell_pos] = gpu_part->w[part_ind];
    gpu_part->cell_q[cell_id * MAX_PART_PER_CELL + cell_pos] = gpu_part->q[part_ind];
}
