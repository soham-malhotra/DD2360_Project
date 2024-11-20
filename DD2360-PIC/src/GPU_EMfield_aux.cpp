#include "GPU_EMfield_aux.h"


void gpuFieldAuxAllocateAndCpy(const struct grid& grd, struct GPUEMfield_aux* gpu_emf_aux, const struct EMfield_aux& emf_aux) {
    
    // define the size of the arrays
    size_t size = grid.nxc*grid.nyc*grid.nzc*sizeof(FPfield);

    // allocate electric field
    cudaErrorHandling(cudaMalloc(&gpu_emf_aux->Phi_flat, size));
    cudaErrorHandling(cudaMemcpy(gpu_emf_aux->Phi_flat, emf_aux.Phi_flat, size, cudaMemcpyHostToDevice));

    cudaErrorHandling(cudaMalloc(&gpu_emf_aux->Exth_flat, size));
    cudaErrorHandling(cudaMemcpy(gpu_emf_aux->Exth_flat, emf_aux.Exth_flat, size, cudaMemcpyHostToDevice));

    cudaErrorHandling(cudaMalloc(&gpu_emf_aux->Eyth_flat, size));
    cudaErrorHandling(cudaMemcpy(gpu_emf_aux->Eyth_flat, emf_aux.Eyth_flat, size, cudaMemcpyHostToDevice));

    cudaErrorHandling(cudaMalloc(&gpu_emf_aux->Ezth_flat, size));
    cudaErrorHandling(cudaMemcpy(gpu_emf_aux->Ezth_flat, emf_aux.Ezth_flat, size, cudaMemcpyHostToDevice));
    
    // allocate magnetic field
    cudaErrorHandling(cudaMalloc(&gpu_emf_aux->Bxc_flat, size));
    cudaErrorHandling(cudaMemcpy(gpu_emf_aux->Bxc_flat, emf_aux.Bxc_flat, size, cudaMemcpyHostToDevice));

    cudaErrorHandling(cudaMalloc(&gpu_emf_aux->Byc_flat, size));
    cudaErrorHandling(cudaMemcpy(gpu_emf_aux->Byc_flat, emf_aux.Byc_flat, size, cudaMemcpyHostToDevice));

    cudaErrorHandling(cudaMalloc(&gpu_emf_aux->Bzc_flat, size));
    cudaErrorHandling(cudaMemcpy(gpu_emf_aux->Bzc_flat, emf_aux.Bzc_flat, size, cudaMemcpyHostToDevice));
    
}

void gpuFieldAuxDeallocate(struct GPUEMfield_aux* gpu_emf_aux) {
    //deallocate electric fields
    cudaErrorHandling(cudaFree(gpu_emf_aux->Phi_flat));
    cudaErrorHandling(cudaFree(gpu_emf_aux->Exth_flat));
    cudaErrorHandling(cudaFree(gpu_emf_aux->Eyth_flat));
    cudaErrorHandling(cudaFree(gpu_emf_aux->Ezth_flat));

    //deallocate magnetic fields
    cudaErrorHandling(cudaFree(gpu_emf_aux->Bxc_flat));
    cudaErrorHandling(cudaFree(gpu_emf_aux->Byc_flat));
    cudaErrorHandling(cudaFree(gpu_emf_aux->Bzc_flat));
}
