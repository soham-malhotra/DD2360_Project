#include "GPU_Grid.h"

struct GPUGrid* gpuGridAllocateAndCpy(const grid& grd) {
    // define field array size
    GPUGrid* gpu_grd = nullptr;
    size_t size = grd.nxn * grd.nyn * grd.nzn * sizeof(FPfield);

    //allocate device memory for the grid
    cudaErrorHandling(cudaMalloc(&gpu_grd, sizeof(GPUGrid)));

    // create a temporary grid on the host
    GPUGrid temp_grid;
    copyStaticMembers(grd, temp_grid);
    cudaErrorHandling(cudaMemcpy(gpu_grd, &temp_grid, sizeof(GPUGrid), cudaMemcpyHostToDevice));
    
    // allocate coordinate nodes on device memory
    copyArrayToDeviceStruct<FPfield>(&(gpu_grd->XN_GPU_flat), grd.XN_flat, size);
    copyArrayToDeviceStruct<FPfield>(&(gpu_grd->YN_GPU_flat), grd.YN_flat, size);
    copyArrayToDeviceStruct<FPfield>(&(gpu_grd->ZN_GPU_flat), grd.ZN_flat, size);

    return gpu_grd;
}

void gpuGridDeallocate(GPUGrid* gpu_grd) {
    // deallocate device memory for the grid coordinate nodes
    GPUGrid temp_grid;
    cudaErrorHandling(cudaMemcpy(&temp_grid, gpu_grd, sizeof(GPUGrid), cudaMemcpyDeviceToHost));

    cudaErrorHandling(cudaFree(temp_grid.XN_GPU_flat));
    cudaErrorHandling(cudaFree(temp_grid.YN_GPU_flat));
    cudaErrorHandling(cudaFree(temp_grid.ZN_GPU_flat));

    // deallocate device memory for the grid
    cudaErrorHandling(cudaFree(gpu_grd));
}

void copyStaticMembers(const grid& grd, GPUGrid& gpu_grd) {
    //copy number of cells and nodes
    gpu_grd.nxc = grd.nxc;
    gpu_grd.nxn = grd.nxn;
    gpu_grd.nyc = grd.nyc;
    gpu_grd.nyn = grd.nyn;
    gpu_grd.nzc = grd.nzc;
    gpu_grd.nzn = grd.nzn; 
        
    //copy space step and inverse of this
    gpu_grd.dx = grd.dx;
    gpu_grd.dy = grd.dy;
    gpu_grd.dz = grd.dz;
    gpu_grd.invdx = grd.invdx;
    gpu_grd.invdy = grd.invdy;
    gpu_grd.invdz = grd.invdz;
    gpu_grd.invVOL = grd.invVOL;

    //copy local grid boundaries coord
    gpu_grd.xStart = grd.xStart;
    gpu_grd.xEnd = grd.xEnd;
    gpu_grd.yStart = grd.yStart;
    gpu_grd.yEnd = grd.yEnd;
    gpu_grd.zStart = grd.zStart;
    gpu_grd.zEnd = grd.zEnd;

    //copy domain size
    gpu_grd.Lx = grd.Lx;
    gpu_grd.Ly = grd.Ly;
    gpu_grd.Lz = grd.Lz;

    //periodicity for the fields
    gpu_grd.PERIODICX = grd.PERIODICX;
    gpu_grd.PERIODICY = grd.PERIODICY;
    gpu_grd.PERIODICZ = grd.PERIODICZ;
}