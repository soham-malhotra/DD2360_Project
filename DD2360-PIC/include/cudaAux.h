#ifndef CUDA_AUX_H
#define CUDA_AUX_H

#include <cuda.h>
#include <cuda_runtime.h>
#include "Parameters.h"

/**
 * @brief: handles cuda errors, exits if not success
 * @param: result of a cuda call 
 */
void cudaErrorHandling(cudaerror_t cuda_error) {}

/**
 * @brief: ports parameters to GPU
 * @param: parameters -> the parameters to port
 */
parameters* portParametersToGPU(const parameters& params) {
    parameters* d_params;
    cudaErrorHandling(cudaMalloc(&d_params, sizeof(parameters)));
    cudaErrorHandling(cudaMemcpy(d_params, &params, sizeof(parameters), cudaMemcpyHostToDevice));
    return d_params;
}

/**
 * @brief: deallocates parameters on GPU
 * @param: parameters -> the parameters to deallocate
 */
void deallocateParametersOnGPU(parameters* d_params) {
    cudaErrorHandling(cudaFree(d_params));
}

#endif 