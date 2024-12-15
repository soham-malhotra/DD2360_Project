#include "GPU_InterpDensSpecies.h"

struct GPUInterpDensSpecies* gpuInterpDensSpeciesAllocateAndCpyStatic(const struct grid& grid, const struct interpDensSpecies& interp_dens_species) {
    GPUInterpDensSpecies* gpu_interp_dens_species = nullptr;

    // define field array size
    size_t size = grid.nxn * grid.nyn * grid.nzn * sizeof(FPinterp);  // number of nodes
    size_t size_c = grid.nxc * grid.nyc * grid.nzc * sizeof(FPinterp); // number of cells

    cudaErrorHandling(cudaMalloc(&gpu_interp_dens_species, sizeof(GPUInterpDensSpecies)));

    // copy static members
    cudaErrorHandling(cudaMemcpy(&gpu_interp_dens_species->species_ID, &interp_dens_species.species_ID, sizeof(interp_dens_species.species_ID), cudaMemcpyHostToDevice));
    
    // allocate densities
    allocateDeviceArray<FPfield>(&gpu_interp_dens_species->rhon_flat, size);
    allocateDeviceArray<FPfield>(&gpu_interp_dens_species->rhoc_flat, size_c);

    // allocate currents
    allocateDeviceArray<FPfield>(&gpu_interp_dens_species->Jx_flat, size);
    allocateDeviceArray<FPfield>(&gpu_interp_dens_species->Jy_flat, size);
    allocateDeviceArray<FPfield>(&gpu_interp_dens_species->Jz_flat, size);

    // allocate pressure tensor
    allocateDeviceArray<FPfield>(&gpu_interp_dens_species->pxx_flat, size);
    allocateDeviceArray<FPfield>(&gpu_interp_dens_species->pxy_flat, size);
    allocateDeviceArray<FPfield>(&gpu_interp_dens_species->pxz_flat, size);
    allocateDeviceArray<FPfield>(&gpu_interp_dens_species->pyy_flat, size);
    allocateDeviceArray<FPfield>(&gpu_interp_dens_species->pyz_flat, size);
    allocateDeviceArray<FPfield>(&gpu_interp_dens_species->pzz_flat, size);

    return gpu_interp_dens_species;
}

void gpuInterpDensSpeciesCpyTo(const struct grid& grid, const struct interpDensSpecies& interp_dens_species, struct GPUInterpDensSpecies* gpu_interp_dens_species) {
    GPUInterpDensSpecies temp_interp_dens_species;
    cudaErrorHandling(cudaMemcpy(&temp_interp_dens_species, gpu_interp_dens_species, sizeof(GPUInterpDensSpecies), cudaMemcpyDeviceToHost));

    // define field array size
    size_t size = grid.nxn * grid.nyn * grid.nzn * sizeof(FPinterp);  // number of nodes
    size_t size_c = grid.nxc * grid.nyc * grid.nzc * sizeof(FPinterp); // number of cells

    // copy densities
    copyArrayToDevice<FPfield>(temp_interp_dens_species.rhon_flat, interp_dens_species.rhon_flat, size);
    copyArrayToDevice<FPfield>(temp_interp_dens_species.rhoc_flat, interp_dens_species.rhoc_flat, size_c);

    // copy currents
    copyArrayToDevice<FPfield>(temp_interp_dens_species.Jx_flat, interp_dens_species.Jx_flat, size);
    copyArrayToDevice<FPfield>(temp_interp_dens_species.Jy_flat, interp_dens_species.Jy_flat, size);
    copyArrayToDevice<FPfield>(temp_interp_dens_species.Jz_flat, interp_dens_species.Jz_flat, size);

    // copy pressure tensor
    copyArrayToDevice<FPfield>(temp_interp_dens_species.pxx_flat, interp_dens_species.pxx_flat, size);
    copyArrayToDevice<FPfield>(temp_interp_dens_species.pxy_flat, interp_dens_species.pxy_flat, size);
    copyArrayToDevice<FPfield>(temp_interp_dens_species.pxz_flat, interp_dens_species.pxz_flat, size);
    copyArrayToDevice<FPfield>(temp_interp_dens_species.pyy_flat, interp_dens_species.pyy_flat, size);
    copyArrayToDevice<FPfield>(temp_interp_dens_species.pyz_flat, interp_dens_species.pyz_flat, size);
    copyArrayToDevice<FPfield>(temp_interp_dens_species.pzz_flat, interp_dens_species.pzz_flat, size);
}

void gpuInterpDensSpeciesCpyBack(const struct grid& grid, struct interpDensSpecies& interp_dens_species, const struct GPUInterpDensSpecies* gpu_interp_dens_species) {
    GPUInterpDensSpecies temp_interp_dens_species;
    cudaErrorHandling(cudaMemcpy(&temp_interp_dens_species, gpu_interp_dens_species, sizeof(GPUInterpDensSpecies), cudaMemcpyDeviceToHost));

    // define field array size
    size_t size = grid.nxn * grid.nyn * grid.nzn * sizeof(FPinterp);  // number of nodes
    size_t size_c = grid.nxc * grid.nyc * grid.nzc * sizeof(FPinterp); // number of cells

    // copy densities
    copyArrayFromDevice<FPfield>(interp_dens_species.rhon_flat, temp_interp_dens_species.rhon_flat, size);
    copyArrayFromDevice<FPfield>(interp_dens_species.rhoc_flat, temp_interp_dens_species.rhoc_flat, size_c);

    // copy currents
    copyArrayFromDevice<FPfield>(interp_dens_species.Jx_flat, temp_interp_dens_species.Jx_flat, size);
    copyArrayFromDevice<FPfield>(interp_dens_species.Jy_flat, temp_interp_dens_species.Jy_flat, size);
    copyArrayFromDevice<FPfield>(interp_dens_species.Jz_flat, temp_interp_dens_species.Jz_flat, size);

    // copy pressure tensor
    copyArrayFromDevice<FPfield>(interp_dens_species.pxx_flat, temp_interp_dens_species.pxx_flat, size);
    copyArrayFromDevice<FPfield>(interp_dens_species.pxy_flat, temp_interp_dens_species.pxy_flat, size);
    copyArrayFromDevice<FPfield>(interp_dens_species.pxz_flat, temp_interp_dens_species.pxz_flat, size);
    copyArrayFromDevice<FPfield>(interp_dens_species.pyy_flat, temp_interp_dens_species.pyy_flat, size);
    copyArrayFromDevice<FPfield>(interp_dens_species.pyz_flat, temp_interp_dens_species.pyz_flat, size);
    copyArrayFromDevice<FPfield>(interp_dens_species.pzz_flat, temp_interp_dens_species.pzz_flat, size);
}

void gpuInterpDensSpeciesDeallocate(struct GPUInterpDensSpecies* gpu_interp_dens_species) {
    GPUInterpDensSpecies temp_interp_dens_species;
    cudaErrorHandling(cudaMemcpy(&temp_interp_dens_species, gpu_interp_dens_species, sizeof(GPUInterpDensSpecies), cudaMemcpyDeviceToHost));

    //deallocate densities
    cudaErrorHandling(cudaFree(temp_interp_dens_species.rhon_flat));
    cudaErrorHandling(cudaFree(temp_interp_dens_species.rhoc_flat));

    //deallocate currents
    cudaErrorHandling(cudaFree(temp_interp_dens_species.Jx_flat));
    cudaErrorHandling(cudaFree(temp_interp_dens_species.Jy_flat));
    cudaErrorHandling(cudaFree(temp_interp_dens_species.Jz_flat));

    //deallocate pressure tensor
    cudaErrorHandling(cudaFree(temp_interp_dens_species.pxx_flat));
    cudaErrorHandling(cudaFree(temp_interp_dens_species.pxy_flat));
    cudaErrorHandling(cudaFree(temp_interp_dens_species.pxz_flat));
    cudaErrorHandling(cudaFree(temp_interp_dens_species.pyy_flat));
    cudaErrorHandling(cudaFree(temp_interp_dens_species.pyz_flat));
    cudaErrorHandling(cudaFree(temp_interp_dens_species.pzz_flat));

    //deallocate the struct itself
    cudaErrorHandling(cudaFree(gpu_interp_dens_species));
}