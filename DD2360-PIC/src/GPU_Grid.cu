#include "GPU_Grid.h"

struct GPUGrid* gpuGridAllocateAndCpy(const grid& grd) {  // TODO maybe just manually cudaMemcpy every field?
    // define field array size
    GPUGrid* gpu_grd = nullptr;
    size_t size = grd.nxn * grd.nyn * grd.nzn * sizeof(FPfield);

    // allocate device memory for the grid
    cudaErrorHandling(cudaMalloc(&gpu_grd, sizeof(GPUGrid)));

    // copy static members
    cudaErrorHandling(cudaMemcpy(&(gpu_grd->nxc), &grd.nxc, sizeof(grd.nxc), cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMemcpy(&(gpu_grd->nxn), &grd.nxn, sizeof(grd.nxn), cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMemcpy(&(gpu_grd->nyc), &grd.nyc, sizeof(grd.nyc), cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMemcpy(&(gpu_grd->nyn), &grd.nyn, sizeof(grd.nyn), cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMemcpy(&(gpu_grd->nzc), &grd.nzc, sizeof(grd.nzc), cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMemcpy(&(gpu_grd->nzn), &grd.nzn, sizeof(grd.nzn), cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMemcpy(&(gpu_grd->dx), &grd.dx, sizeof(grd.dx), cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMemcpy(&(gpu_grd->dy), &grd.dy, sizeof(grd.dy), cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMemcpy(&(gpu_grd->dz), &grd.dz, sizeof(grd.dz), cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMemcpy(&(gpu_grd->invdx), &grd.invdx, sizeof(grd.invdx), cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMemcpy(&(gpu_grd->invdy), &grd.invdy, sizeof(grd.invdy), cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMemcpy(&(gpu_grd->invdz), &grd.invdz, sizeof(grd.invdz), cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMemcpy(&(gpu_grd->invVOL), &grd.invVOL, sizeof(grd.invVOL), cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMemcpy(&(gpu_grd->xStart), &grd.xStart, sizeof(grd.xStart), cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMemcpy(&(gpu_grd->xEnd), &grd.xEnd, sizeof(grd.xEnd), cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMemcpy(&(gpu_grd->yStart), &grd.yStart, sizeof(grd.yStart), cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMemcpy(&(gpu_grd->yEnd), &grd.yEnd, sizeof(grd.yEnd), cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMemcpy(&(gpu_grd->zStart), &grd.zStart, sizeof(grd.zStart), cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMemcpy(&(gpu_grd->zEnd), &grd.zEnd, sizeof(grd.zEnd), cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMemcpy(&(gpu_grd->Lx), &grd.Lx, sizeof(grd.Lx), cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMemcpy(&(gpu_grd->Ly), &grd.Ly, sizeof(grd.Ly), cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMemcpy(&(gpu_grd->Lz), &grd.Lz, sizeof(grd.Lz), cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMemcpy(&(gpu_grd->PERIODICX), &grd.PERIODICX, sizeof(grd.PERIODICX), cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMemcpy(&(gpu_grd->PERIODICY), &grd.PERIODICY, sizeof(grd.PERIODICY), cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMemcpy(&(gpu_grd->PERIODICZ), &grd.PERIODICZ, sizeof(grd.PERIODICZ), cudaMemcpyHostToDevice));

    // allocate coordinate nodes on device memory
    allocateAndCpyDeviceArray<FPfield>(&(gpu_grd->XN_GPU_flat), grd.XN_flat, size);
    allocateAndCpyDeviceArray<FPfield>(&(gpu_grd->YN_GPU_flat), grd.YN_flat, size);
    allocateAndCpyDeviceArray<FPfield>(&(gpu_grd->ZN_GPU_flat), grd.ZN_flat, size);

    return gpu_grd;
}

void gpuGridDeallocate(GPUGrid* gpu_grd) {
    // deallocate device memory for the grid coordinate nodes
    GPUGrid temp_grd;
    cudaErrorHandling(cudaMemcpy(&temp_grd, gpu_grd, sizeof(GPUGrid), cudaMemcpyDeviceToHost));

    cudaErrorHandling(cudaFree(temp_grd.XN_GPU_flat));
    cudaErrorHandling(cudaFree(temp_grd.YN_GPU_flat));
    cudaErrorHandling(cudaFree(temp_grd.ZN_GPU_flat));

    // deallocate device memory for the grid
    cudaErrorHandling(cudaFree(gpu_grd));
}
