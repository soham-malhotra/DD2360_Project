#include "GPU_InterpDensNet.h"

void gpuInterpDensNetAllocateAndCpy(const struct grid& grid, struct GPUInterpDensNet* gpu_interp_dens_net, const struct interpDensNet& interp_dens_net) {
    // define field array size
    size_t size = grid.nxn * grid.nyn * grid.nzn * sizeof(FPinterp);  // number of nodes
    size_t size_c = grid.nxc * grid.nyc * grid.nzc * sizeof(FPinterp); // number of cells

    cudaErrorHandling(cudaMalloc(&gpu_interp_dens_net, sizeof(interpDensNet)));
    // nothing to copy

    // allocate densities
    cudaErrorHandling(cudaMalloc(&gpu_interp_dens_net->rhon_flat, size));
    cudaErrorHandling(cudaMemcpy(gpu_interp_dens_net->rhon_flat, interp_dens_net.rhon_flat, size, cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMalloc(&gpu_interp_dens_net->rhoc_flat, size_c));
    cudaErrorHandling(cudaMemcpy(gpu_interp_dens_net->rhoc_flat, interp_dens_net.rhoc_flat, size_c, cudaMemcpyHostToDevice));

    // allocate currents
    cudaErrorHandling(cudaMalloc(&gpu_interp_dens_net->Jx_flat, size));
    cudaErrorHandling(cudaMemcpy(gpu_interp_dens_net->Jx_flat, interp_dens_net.Jx_flat, size, cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMalloc(&gpu_interp_dens_net->Jy_flat, size));
    cudaErrorHandling(cudaMemcpy(gpu_interp_dens_net->Jy_flat, interp_dens_net.Jy_flat, size, cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMalloc(&gpu_interp_dens_net->Jz_flat, size));
    cudaErrorHandling(cudaMemcpy(gpu_interp_dens_net->Jz_flat, interp_dens_net.Jz_flat, size, cudaMemcpyHostToDevice));

    // allocate pressure tensor
    cudaErrorHandling(cudaMalloc(&gpu_interp_dens_net->pxx_flat, size));
    cudaErrorHandling(cudaMemcpy(gpu_interp_dens_net->pxx_flat, interp_dens_net.pxx_flat, size, cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMalloc(&gpu_interp_dens_net->pxy_flat, size));
    cudaErrorHandling(cudaMemcpy(gpu_interp_dens_net->pxy_flat, interp_dens_net.pxy_flat, size, cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMalloc(&gpu_interp_dens_net->pxz_flat, size));
    cudaErrorHandling(cudaMemcpy(gpu_interp_dens_net->pxz_flat, interp_dens_net.pxz_flat, size, cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMalloc(&gpu_interp_dens_net->pyy_flat, size));
    cudaErrorHandling(cudaMemcpy(gpu_interp_dens_net->pyy_flat, interp_dens_net.pyy_flat, size, cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMalloc(&gpu_interp_dens_net->pyz_flat, size));
    cudaErrorHandling(cudaMemcpy(gpu_interp_dens_net->pyz_flat, interp_dens_net.pyz_flat, size, cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMalloc(&gpu_interp_dens_net->pzz_flat, size));
    cudaErrorHandling(cudaMemcpy(gpu_interp_dens_net->pzz_flat, interp_dens_net.pzz_flat, size, cudaMemcpyHostToDevice));
}