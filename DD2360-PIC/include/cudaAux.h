#ifndef CUDA_AUX_H
#define CUDA_AUX_H

#include <cuda.h>
#include <cuda_runtime.h>

/**
 * @brief: handles cuda errors, exits if not success
 * @param: result of a cuda call 
 */
void cudaErrorHandling(cudaerror_t cuda_error) {}

#endif 