#include "GPU_Grid.h"

void gpuGridAllocateAndCpy(const grid& grid, GPUgrid* gpu_grid) {
    // define field array size
    size_t size = grid.nxn * grid.nyn * grid.nzn * sizeof(FPfield);

    cudaErrorHandling(cudaMalloc(&gpu_grid, sizeof(grid)));
    cudaMemcpy(gpu_grid, &grid, sizeof(grid), cudaMemcpyHostToDevice);

    // allocate coordinate node X
    cudaErrorHandling(cudaMalloc(&gpu_grid->XN_GPU_flat, size));  // should overwrite old pointer
    cudaErrorHandling(cudaMemcpy(gpu_grid->XN_GPU_flat, grid.XN_flat, size, cudaMemcpyHostToDevice));

    // allocate coordinate node Y
    cudaErrorHandling(cudaMalloc(&gpu_grid->YN_GPU_flat, size));
    cudaErrorHandling(cudaMemcpy(gpu_grid->YN_GPU_flat, grid.YN_flat, size, cudaMemcpyHostToDevice));

    // allocate coordinate node Z
    cudaErrorHandling(cudaMalloc(&gpu_grid->ZN_GPU_flat, size));
    cudaErrorHandling(cudaMemcpy(gpu_grid->ZN_GPU_flat, grid.ZN_flat, size, cudaMemcpyHostToDevice));
}

void gpuGridDeallocate(GPUgrid* gpu_grid) {
    cudaErrorHandling(cudaFree(gpu_grid->XN_GPU_flat));
    cudaErrorHandling(cudaFree(gpu_grid->YN_GPU_flat));
    cudaErrorHandling(cudaFree(gpu_grid->ZN_GPU_flat));
}