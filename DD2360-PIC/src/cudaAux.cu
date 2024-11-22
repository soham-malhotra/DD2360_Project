#include "cudaAux.h"
#include <cstdio>
#include <cstdlib>

void cudaErrorHandling(cudaError_t cuda_error) {
    if(cuda_error != cudaSuccess) {
       std::cerr << "CUDA Error: " << cudaGetErrorString(cuda_error) << std::endl;
        exit(EXIT_FAILURE);  // Or handle the error as needed
    }
}

void copyArrayToDeviceStruct(FP** struct_device_array, FP* host_array, size_t size) {
    FP* temp_device_array;
    cudaErrorHandling(cudaMalloc(&temp_device_array, size));
    cudaErrorHandling(cudaMemcpy(temp_device_array, host_array, size, cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMemcpy(struct_device_array, &temp_device_array, sizeof(FP*), cudaMemcpyHostToDevice)); // copy device address into device struct
}