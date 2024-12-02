#ifndef CUDA_AUX_H
#define CUDA_AUX_H

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

/**
 * @brief: handles cuda errors, exits if not success
 * @param: result of a cuda call 
 */
inline void cudaErrorHandling(cudaError_t cuda_error, std::string message="") {
    if(cuda_error != cudaSuccess) {
       std::cerr << "CUDA Error: " << cudaGetErrorString(cuda_error) << ". Message: " << message << std::endl;
       throw std::runtime_error("CUDA Error");
    }
}

template <class FP>
inline void allocateAndCpyDeviceArray(FP** struct_device_array, FP* host_array, size_t size) {
    FP* temp_device_array;
    cudaErrorHandling(cudaMalloc(&temp_device_array, size));
    cudaErrorHandling(cudaMemcpy(temp_device_array, host_array, size, cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMemcpy(struct_device_array, &temp_device_array, sizeof(FP*), cudaMemcpyHostToDevice)); // copy device address into device struct
}

/**
 * @brief: allocates memory on device and copies data from host to device
 * @param: device_array: pointer to array on device memory
 * @param: size: size of memory to allocate
 */
template <class FP>
inline void allocateDeviceArray(FP** struct_device_array, size_t size) {
    FP* temp_device_array;
    cudaErrorHandling(cudaMalloc(&temp_device_array, size));  // make a pointer to a block in device memory
    cudaErrorHandling(cudaMemcpy(struct_device_array, &temp_device_array, sizeof(FP*), cudaMemcpyHostToDevice)); // copy device address into device struct
}

template <class FP>
inline void copyArrayToDevice(FP* device_array, FP* host_array, size_t size) {
    cudaErrorHandling(cudaMemcpy(device_array, host_array, size, cudaMemcpyHostToDevice));
}

template <class FP>
inline void copyArrayFromDevice(FP* host_array, FP* device_array, size_t size) {
    cudaErrorHandling(cudaMemcpy(host_array, device_array, size, cudaMemcpyDeviceToHost));
}

#endif