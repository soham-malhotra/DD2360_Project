#include "GPU_Parameters.h"

void gpuParametersAllocateAndCpy(struct GPUParameters* gpu_param, const struct parameters& param) {

    cudaErrorHandling(cudaMalloc(&gpu_param, sizeof(parameters)));
    cudaErrorHandling(cudaMemcpy(gpu_param, &param, sizeof(parameters), cudaMemcpyHostToDevice));
}

void gpuParametersDeallocate(struct GPUParameters* gpu_param) {
    cudaErrorHandling(cudaFree(gpu_param));
}