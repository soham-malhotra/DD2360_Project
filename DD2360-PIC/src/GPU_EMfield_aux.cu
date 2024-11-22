#include "GPU_EMfield_aux.h"


void gpuFieldAuxAllocateAndCpy(const struct grid& grid, struct GPUEMfield_aux* gpu_field_aux, const struct EMfield_aux& field_aux) {
    
    // define the size of the arrays
    size_t size = grid.nxc * grid.nyc * grid.nzc * sizeof(FPfield);

    cudaErrorHandling(cudaMalloc(&gpu_field_aux, sizeof(struct GPUEMfield_aux)));
    // nothing to copy

    // allocate electric field
    copyArrayToDeviceStruct<FPfield>(&(gpu_field_aux->Phi_flat), field_aux.Phi_flat, size);

    copyArrayToDeviceStruct<FPfield>(&(gpu_field_aux->Exth_flat), field_aux.Exth_flat, size);

    copyArrayToDeviceStruct<FPfield>(&(gpu_field_aux->Eyth_flat), field_aux.Eyth_flat, size);

    copyArrayToDeviceStruct<FPfield>(&(gpu_field_aux->Ezth_flat), field_aux.Ezth_flat, size);
    
    // allocate magnetic field
    copyArrayToDeviceStruct<FPfield>(&(gpu_field_aux->Bxc_flat), field_aux.Bxc_flat, size);

    copyArrayToDeviceStruct<FPfield>(&(gpu_field_aux->Byc_flat), field_aux.Byc_flat, size);

    copyArrayToDeviceStruct<FPfield>(&(gpu_field_aux->Bzc_flat), field_aux.Bzc_flat, size);
    
}

void gpuFieldAuxDeallocate(struct GPUEMfield_aux* gpu_field_aux) {

    //deallocate electric fields
    cudaErrorHandling(cudaFree(gpu_field_aux->Phi_flat));
    cudaErrorHandling(cudaFree(gpu_field_aux->Exth_flat));
    cudaErrorHandling(cudaFree(gpu_field_aux->Eyth_flat));
    cudaErrorHandling(cudaFree(gpu_field_aux->Ezth_flat));

    //deallocate magnetic fields
    cudaErrorHandling(cudaFree(gpu_field_aux->Bxc_flat));
    cudaErrorHandling(cudaFree(gpu_field_aux->Byc_flat));
    cudaErrorHandling(cudaFree(gpu_field_aux->Bzc_flat));

    cudaErrorHandling(cudaFree(gpu_field_aux));
}
