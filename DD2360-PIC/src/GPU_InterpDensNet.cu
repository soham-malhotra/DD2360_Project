#include "GPU_InterpDensNet.h"

struct GPUInterpDensNet* gpuInterpDensNetAllocate(const struct grid& grid) {
    GPUInterpDensNet* gpu_interp_dens_net = nullptr;

    // define field array size
    size_t size = grid.nxn * grid.nyn * grid.nzn * sizeof(FPinterp);  // number of nodes
    size_t size_c = grid.nxc * grid.nyc * grid.nzc * sizeof(FPinterp); // number of cells

    // allocate device memory for the grid
    cudaErrorHandling(cudaMalloc(&gpu_interp_dens_net, sizeof(GPUInterpDensNet)));

    // allocate densities
    allocateDeviceArray<FPfield>(&gpu_interp_dens_net->rhon_flat, size);
    allocateDeviceArray<FPfield>(&gpu_interp_dens_net->rhoc_flat, size_c);

    // allocate currents
    allocateDeviceArray<FPfield>(&gpu_interp_dens_net->Jx_flat, size);
    allocateDeviceArray<FPfield>(&gpu_interp_dens_net->Jy_flat, size);
    allocateDeviceArray<FPfield>(&gpu_interp_dens_net->Jz_flat, size);

    // allocate pressure tensor
    allocateDeviceArray<FPfield>(&gpu_interp_dens_net->pxx_flat, size);
    allocateDeviceArray<FPfield>(&gpu_interp_dens_net->pxy_flat, size);
    allocateDeviceArray<FPfield>(&gpu_interp_dens_net->pxz_flat, size);
    allocateDeviceArray<FPfield>(&gpu_interp_dens_net->pyy_flat, size);
    allocateDeviceArray<FPfield>(&gpu_interp_dens_net->pyz_flat, size);
    allocateDeviceArray<FPfield>(&gpu_interp_dens_net->pzz_flat, size);

    return gpu_interp_dens_net;
}

void gpuInterpDensNetCpyTo(const struct grid& grid, const struct interpDensNet& interp_dens_net, struct GPUInterpDensNet* gpu_interp_dens_net) {
    GPUInterpDensNet temp_interp_dens_net;
    cudaErrorHandling(cudaMemcpy(&temp_interp_dens_net, gpu_interp_dens_net, sizeof(GPUInterpDensNet), cudaMemcpyDeviceToHost));

    // define field array size
    size_t size = grid.nxn * grid.nyn * grid.nzn * sizeof(FPinterp);  // number of nodes
    size_t size_c = grid.nxc * grid.nyc * grid.nzc * sizeof(FPinterp); // number of cells

    // copy densities
    copyArrayToDevice<FPfield>(temp_interp_dens_net.rhon_flat, interp_dens_net.rhon_flat, size);
    copyArrayToDevice<FPfield>(temp_interp_dens_net.rhoc_flat, interp_dens_net.rhoc_flat, size_c);

    // copy currents
    copyArrayToDevice<FPfield>(temp_interp_dens_net.Jx_flat, interp_dens_net.Jx_flat, size);
    copyArrayToDevice<FPfield>(temp_interp_dens_net.Jy_flat, interp_dens_net.Jy_flat, size);
    copyArrayToDevice<FPfield>(temp_interp_dens_net.Jz_flat, interp_dens_net.Jz_flat, size);

    // copy pressure tensor
    copyArrayToDevice<FPfield>(temp_interp_dens_net.pxx_flat, interp_dens_net.pxx_flat, size);
    copyArrayToDevice<FPfield>(temp_interp_dens_net.pxy_flat, interp_dens_net.pxy_flat, size);
    copyArrayToDevice<FPfield>(temp_interp_dens_net.pxz_flat, interp_dens_net.pxz_flat, size);
    copyArrayToDevice<FPfield>(temp_interp_dens_net.pyy_flat, interp_dens_net.pyy_flat, size);
    copyArrayToDevice<FPfield>(temp_interp_dens_net.pyz_flat, interp_dens_net.pyz_flat, size);
    copyArrayToDevice<FPfield>(temp_interp_dens_net.pzz_flat, interp_dens_net.pzz_flat, size);
}

void gpuInterpDensNetCpyBack(const struct grid& grid, struct interpDensNet& interp_dens_net, const struct GPUInterpDensNet* gpu_interp_dens_net) {
    GPUInterpDensNet temp_interp_dens_net;
    cudaErrorHandling(cudaMemcpy(&temp_interp_dens_net, gpu_interp_dens_net, sizeof(GPUInterpDensNet), cudaMemcpyDeviceToHost));

    // define field array size
    size_t size = grid.nxn * grid.nyn * grid.nzn * sizeof(FPinterp);  // number of nodes
    size_t size_c = grid.nxc * grid.nyc * grid.nzc * sizeof(FPinterp); // number of cells

    // copy densities
    copyArrayFromDevice<FPfield>(interp_dens_net.rhon_flat, temp_interp_dens_net.rhon_flat, size);
    copyArrayFromDevice<FPfield>(interp_dens_net.rhoc_flat, temp_interp_dens_net.rhoc_flat, size_c);

    // copy currents
    copyArrayFromDevice<FPfield>(interp_dens_net.Jx_flat, temp_interp_dens_net.Jx_flat, size);
    copyArrayFromDevice<FPfield>(interp_dens_net.Jy_flat, temp_interp_dens_net.Jy_flat, size);
    copyArrayFromDevice<FPfield>(interp_dens_net.Jz_flat, temp_interp_dens_net.Jz_flat, size);

    // copy pressure tensor
    copyArrayFromDevice<FPfield>(interp_dens_net.pxx_flat, temp_interp_dens_net.pxx_flat, size);
    copyArrayFromDevice<FPfield>(interp_dens_net.pxy_flat, temp_interp_dens_net.pxy_flat, size);
    copyArrayFromDevice<FPfield>(interp_dens_net.pxz_flat, temp_interp_dens_net.pxz_flat, size);
    copyArrayFromDevice<FPfield>(interp_dens_net.pyy_flat, temp_interp_dens_net.pyy_flat, size);
    copyArrayFromDevice<FPfield>(interp_dens_net.pyz_flat, temp_interp_dens_net.pyz_flat, size);
    copyArrayFromDevice<FPfield>(interp_dens_net.pzz_flat, temp_interp_dens_net.pzz_flat, size);
}

void gpuInterpDensNetDeallocate(struct GPUInterpDensNet* gpu_interp_dens_net) {
    GPUInterpDensNet temp_interp_dens_net;
    cudaErrorHandling(cudaMemcpy(&temp_interp_dens_net, gpu_interp_dens_net, sizeof(GPUInterpDensNet), cudaMemcpyDeviceToHost));

    //deallocate densities
    cudaErrorHandling(cudaFree(temp_interp_dens_net.rhon_flat));
    cudaErrorHandling(cudaFree(temp_interp_dens_net.rhoc_flat));

    //deallocate currents
    cudaErrorHandling(cudaFree(temp_interp_dens_net.Jx_flat));
    cudaErrorHandling(cudaFree(temp_interp_dens_net.Jy_flat));
    cudaErrorHandling(cudaFree(temp_interp_dens_net.Jz_flat));

    //deallocate pressure tensor
    cudaErrorHandling(cudaFree(temp_interp_dens_net.pxx_flat));
    cudaErrorHandling(cudaFree(temp_interp_dens_net.pxy_flat));
    cudaErrorHandling(cudaFree(temp_interp_dens_net.pxz_flat));
    cudaErrorHandling(cudaFree(temp_interp_dens_net.pyy_flat));
    cudaErrorHandling(cudaFree(temp_interp_dens_net.pyz_flat));
    cudaErrorHandling(cudaFree(temp_interp_dens_net.pzz_flat));

    //deallocate device memory for the grid
    cudaErrorHandling(cudaFree(gpu_interp_dens_net));
}