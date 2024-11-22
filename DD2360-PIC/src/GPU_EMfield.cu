#include "GPU_EMfield.h"

void gpuFieldAllocateAndCpy(const struct grid& grid, struct GPUEMfield* gpu_em_field, const struct EMfield& em_field) {
    
    // define field array size
    size_t size = grid.nxn * grid.nyn * grid.nzn * sizeof(FPfield);

    cudaErrorHandling(cudaMalloc(&gpu_em_field, sizeof(struct GPUEMfield)));
    // nothing to copy

    // allocate electric field
    copyArrayToDeviceStruct<FPfield>(&(gpu_em_field->Ex_flat), em_field.Ex_flat, size);

    copyArrayToDeviceStruct<FPfield>(&(gpu_em_field->Ey_flat), em_field.Ey_flat, size);

    copyArrayToDeviceStruct<FPfield>(&(gpu_em_field->Ez_flat), em_field.Ez_flat, size);

    // allocate magnetic field
    copyArrayToDeviceStruct<FPfield>(&(gpu_em_field->Bxn_flat), em_field.Bxn_flat, size);

    copyArrayToDeviceStruct<FPfield>(&(gpu_em_field->Byn_flat), em_field.Byn_flat, size);

    copyArrayToDeviceStruct<FPfield>(&(gpu_em_field->Bzn_flat), em_field.Bzn_flat, size);

}


void gpuFieldDeallocate(struct GPUEMfield* gpu_field) {
    cudaErrorHandling(cudaFree(gpu_field->Ex_flat));
    cudaErrorHandling(cudaFree(gpu_field->Ey_flat));
    cudaErrorHandling(cudaFree(gpu_field->Ez_flat));
    cudaErrorHandling(cudaFree(gpu_field->Bxn_flat));
    cudaErrorHandling(cudaFree(gpu_field->Byn_flat));
    cudaErrorHandling(cudaFree(gpu_field->Bzn_flat));

    cudaErrorHandling(cudaFree(gpu_field));
}