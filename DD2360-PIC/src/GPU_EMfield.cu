#include "GPU_EMfield.h"

struct GPUEMfield* gpuFieldAllocate(const struct grid& grid) {
    GPUEMfield* gpu_em_field = nullptr;

    // define field array size
    size_t size = grid.nxn * grid.nyn * grid.nzn * sizeof(FPfield);

    // allocate field struct
    cudaErrorHandling(cudaMalloc(&gpu_em_field, sizeof(struct GPUEMfield)));

    // allocate electric field
    allocateDeviceArray<FPfield>(&gpu_em_field->Ex_flat, size);
    allocateDeviceArray<FPfield>(&gpu_em_field->Ey_flat, size);
    allocateDeviceArray<FPfield>(&gpu_em_field->Ez_flat, size);

    // allocate magnetic field
    allocateDeviceArray<FPfield>(&gpu_em_field->Bxn_flat, size);
    allocateDeviceArray<FPfield>(&gpu_em_field->Byn_flat, size);
    allocateDeviceArray<FPfield>(&gpu_em_field->Bzn_flat, size);

    return gpu_em_field;
}

void gpuFieldCpyTo(const struct grid& grid, const struct EMfield& em_field, struct GPUEMfield* gpu_field) {
    GPUEMfield temp_field;
    cudaErrorHandling(cudaMemcpy(&temp_field, gpu_field, sizeof(GPUEMfield), cudaMemcpyDeviceToHost));

    // define field array size
    size_t size = grid.nxn * grid.nyn * grid.nzn * sizeof(FPfield);

    // copy electric field
    copyArrayToDevice<FPfield>(temp_field.Ex_flat, em_field.Ex_flat, size);
    copyArrayToDevice<FPfield>(temp_field.Ey_flat, em_field.Ey_flat, size);
    copyArrayToDevice<FPfield>(temp_field.Ez_flat, em_field.Ez_flat, size);

    // copy magnetic field
    copyArrayToDevice<FPfield>(temp_field.Bxn_flat, em_field.Bxn_flat, size);
    copyArrayToDevice<FPfield>(temp_field.Byn_flat, em_field.Byn_flat, size);
    copyArrayToDevice<FPfield>(temp_field.Bzn_flat, em_field.Bzn_flat, size);
}

void gpuFieldCpyBack(const struct grid& grid, struct EMfield& em_field, const struct GPUEMfield* gpu_field) {
    GPUEMfield temp_field;
    cudaErrorHandling(cudaMemcpy(&temp_field, gpu_field, sizeof(GPUEMfield), cudaMemcpyDeviceToHost));

    // define field array size
    size_t size = grid.nxn * grid.nyn * grid.nzn * sizeof(FPfield);

    // copy electric field
    copyArrayFromDevice<FPfield>(em_field.Ex_flat, temp_field.Ex_flat, size);
    copyArrayFromDevice<FPfield>(em_field.Ey_flat, temp_field.Ey_flat, size);
    copyArrayFromDevice<FPfield>(em_field.Ez_flat, temp_field.Ez_flat, size);

    // copy magnetic field
    copyArrayFromDevice<FPfield>(em_field.Bxn_flat, temp_field.Bxn_flat, size);
    copyArrayFromDevice<FPfield>(em_field.Byn_flat, temp_field.Byn_flat, size);
    copyArrayFromDevice<FPfield>(em_field.Bzn_flat, temp_field.Bzn_flat, size);
}


void gpuFieldDeallocate(struct GPUEMfield* gpu_field) {
    //deallocate electric field
    GPUEMfield temp_field;
    cudaErrorHandling(cudaMemcpy(&temp_field, gpu_field, sizeof(GPUEMfield), cudaMemcpyDeviceToHost));

    cudaErrorHandling(cudaFree(temp_field.Ex_flat));
    cudaErrorHandling(cudaFree(temp_field.Ey_flat));
    cudaErrorHandling(cudaFree(temp_field.Ez_flat));

    //deallocate magnetic field
    cudaErrorHandling(cudaFree(temp_field.Bxn_flat));
    cudaErrorHandling(cudaFree(temp_field.Byn_flat));
    cudaErrorHandling(cudaFree(temp_field.Bzn_flat));

    //deallocate field struct
    cudaErrorHandling(cudaFree(gpu_field));
}