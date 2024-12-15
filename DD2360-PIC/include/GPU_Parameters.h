#ifndef GPU_PARAMETERS_H
#define GPU_PARAMETERS_H

#include "Parameters.h"
#include "cudaAux.h"

// structs are the same
struct GPUParameters : parameters{};

/**
 * @brief: ports parameters to GPU
 * @param: parameters -> the parameters to port
 */
struct GPUParameters* gpuParametersAllocateAndCpy(const struct parameters& param);

/**
 * @brief: deallocates parameters on GPU
 * @param: parameters -> the parameters to deallocate
 */
void gpuParametersDeallocate(struct GPUParameters*);

#endif