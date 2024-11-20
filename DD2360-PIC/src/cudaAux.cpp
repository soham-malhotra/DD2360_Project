#include "cudaAux.h"

void cudaErrorHandling(cudaerror_t cuda_error) {
    if(cuda_error != cudaSuccess) {
        printf("Error in cuda operation!");
        exit(1);
    }
}