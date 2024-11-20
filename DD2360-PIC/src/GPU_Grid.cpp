#include "GPU_Grid.h"

void deallocateGPUGrid(GPUgrid* gpu_grid) {
    cudaErrorHandling(cudaFree(gpu_grid->XN_GPU_flat));
    cudaErrorHandling(cudaFree(gpu_grid->YN_GPU_flat));
    cudaErrorHandling(cudaFree(gpu_grid->ZN_GPU_flat));
}