#include "GPU_Parameters.h"

void gpuParametersAllocateAndCpy(struct GPUParameters* gpu_param, const struct parameters& param) {
    //copy parameters to GPU, no modifications as no dynamic memory allocation
    cudaErrorHandling(cudaMalloc(&gpu_param, sizeof(parameters)));
    cudaErrorHandling(cudaMemcpy(gpu_param, &param, sizeof(parameters), cudaMemcpyHostToDevice));
}

void gpuParametersDeallocate(struct GPUParameters* gpu_param) {
    //deallocate parameters
    cudaErrorHandling(cudaFree(gpu_param));
}