#ifndef CUDA_AUX_H
#define CUDA_AUX_H

#include <cuda_runtime.h>

/**
 * @brief: handles cuda errors, exits if not success
 * @param: result of a cuda call 
 */
void cudaErrorHandling(cudaError_t cuda_error);


/**
 * @brief: allocates memory on device and copies data from host to device
 * @param: device_array: pointer the pointer that points to array on device memory
 * @param: host_array: pointer to host memory
 * @param: size: size of memory to allocate
 */
template <class FP>
void copyArrayToDeviceStruct(FP** struct_device_array, FP* host_array, size_t size) {}

#endif