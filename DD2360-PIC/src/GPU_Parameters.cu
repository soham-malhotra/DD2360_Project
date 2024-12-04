#include "GPU_Parameters.h"

struct GPUParameters* gpuParametersAllocateAndCpy(const struct parameters& param) {

    struct GPUParameters* gpu_param;

    //copy parameters to GPU, no modifications as no dynamic memory allocation
    cudaErrorHandling(cudaMalloc(&gpu_param, sizeof(parameters)));
    cudaErrorHandling(cudaMemcpy(gpu_param, &param, sizeof(parameters), cudaMemcpyHostToDevice));

    return gpu_param;
}

void gpuParametersDeallocate(struct GPUParameters* gpu_param) {
    //deallocate parameters
    cudaErrorHandling(cudaFree(gpu_param));
}