#ifndef CUDA_AUX_H
#define CUDA_AUX_H

#include <cuda_runtime.h>

/**
 * @brief: handles cuda errors, exits if not success
 * @param: result of a cuda call 
 */
void cudaErrorHandling(cudaError_t cuda_error);

template <class FP>
void copyArrayToDeviceStruct(FP** struct_device_array, FP* host_array, size_t size) {
    FP* temp_device_array;
    cudaErrorHandling(cudaMalloc(&temp_device_array, size));
    cudaErrorHandling(cudaMemcpy(temp_device_array, host_array, size, cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMemcpy(struct_device_array, &temp_device_array, sizeof(FP*), cudaMemcpyHostToDevice)); // copy address into struct
}

#endif