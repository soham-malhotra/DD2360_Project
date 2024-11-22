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
void gpuParametersAllocateAndCpy(struct GPUParameters*, const struct parameters&);

/**
 * @brief: deallocates parameters on GPU
 * @param: parameters -> the parameters to deallocate
 */
void gpuParametersDeallocate(struct GPUParameters*);

#endif