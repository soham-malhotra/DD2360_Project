#include "GPU_InterpDensSpecies.h"

void gpuInterpDensSpeciesAllocateAndCpy(const struct grid& grid, struct GPUInterpDensSpecies* gpu_interp_dens_species, const struct interpDensSpecies& interp_dens_species) {
    // define field array size
    size_t size = grid.nxn * grid.nyn * grid.nzn * sizeof(FPinterp);  // number of nodes
    size_t size_c = grid.nxc * grid.nyc * grid.nzc * sizeof(FPinterp); // number of cells

    cudaErrorHandling(cudaMalloc(&gpu_interp_dens_species, sizeof(interpDensSpecies)));
    cudaErrorHandling(cudaMemcpy(gpu_interp_dens_species, &interp_dens_species, sizeof(interpDensSpecies), cudaMemcpyHostToDevice));
    // should include species_ID

    // allocate densities
    cudaErrorHandling(cudaMalloc(&gpu_interp_dens_species->rhon_flat, size));
    cudaErrorHandling(cudaMemcpy(gpu_interp_dens_species->rhon_flat, interp_dens_species.rhon_flat, size, cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMalloc(&gpu_interp_dens_species->rhoc_flat, size_c));
    cudaErrorHandling(cudaMemcpy(gpu_interp_dens_species->rhoc_flat, interp_dens_species.rhoc_flat, size_c, cudaMemcpyHostToDevice));

    // allocate currents
    cudaErrorHandling(cudaMalloc(&gpu_interp_dens_species->Jx_flat, size));
    cudaErrorHandling(cudaMemcpy(gpu_interp_dens_species->Jx_flat, interp_dens_species.Jx_flat, size, cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMalloc(&gpu_interp_dens_species->Jy_flat, size));
    cudaErrorHandling(cudaMemcpy(gpu_interp_dens_species->Jy_flat, interp_dens_species.Jy_flat, size, cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMalloc(&gpu_interp_dens_species->Jz_flat, size));
    cudaErrorHandling(cudaMemcpy(gpu_interp_dens_species->Jz_flat, interp_dens_species.Jz_flat, size, cudaMemcpyHostToDevice));

    // allocate pressure tensor
    cudaErrorHandling(cudaMalloc(&gpu_interp_dens_species->pxx_flat, size));
    cudaErrorHandling(cudaMemcpy(gpu_interp_dens_species->pxx_flat, interp_dens_species.pxx_flat, size, cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMalloc(&gpu_interp_dens_species->pxy_flat, size));
    cudaErrorHandling(cudaMemcpy(gpu_interp_dens_species->pxy_flat, interp_dens_species.pxy_flat, size, cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMalloc(&gpu_interp_dens_species->pxz_flat, size));
    cudaErrorHandling(cudaMemcpy(gpu_interp_dens_species->pxz_flat, interp_dens_species.pxz_flat, size, cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMalloc(&gpu_interp_dens_species->pyy_flat, size));
    cudaErrorHandling(cudaMemcpy(gpu_interp_dens_species->pyy_flat, interp_dens_species.pyy_flat, size, cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMalloc(&gpu_interp_dens_species->pyz_flat, size));
    cudaErrorHandling(cudaMemcpy(gpu_interp_dens_species->pyz_flat, interp_dens_species.pyz_flat, size, cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMalloc(&gpu_interp_dens_species->pzz_flat, size));
    cudaErrorHandling(cudaMemcpy(gpu_interp_dens_species->pzz_flat, interp_dens_species.pzz_flat, size, cudaMemcpyHostToDevice));
}