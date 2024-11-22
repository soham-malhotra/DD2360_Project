#include "cudaAux.h"
#include <cstdio>
#include <cstdlib>

void cudaErrorHandling(cudaError_t cuda_error) {
    if(cuda_error != cudaSuccess) {
        printf("Error in CUDA operation: %s\n", cudaGetErrorString(cuda_error));
        exit(1);
    }
}