#ifndef GPU_INTERPDENSSPECIES_H
#define GPU_INTERPDENSSPECIES_H

#include "Alloc.h"
#include "Grid.h"
#include "InterpDensSpecies.h"
#include "cudaAux.h"

struct GPUInterpDensSpecies {
    
    /** species ID: 0, 1, 2 , ... */
    int species_ID;
    
    // index 1: rho
    FPinterp *rhon_flat;
    FPinterp *rhoc_flat;
    
    // index 2, 3, 4
    FPinterp *Jx_flat;
    FPinterp *Jy_flat;
    FPinterp *Jz_flat;
    // index 5, 6, 7, 8, 9, 10: pressure tensor (symmetric)
    FPinterp *pxx_flat;
    FPinterp *pxy_flat;
    FPinterp *pxz_flat;
    FPinterp *pyy_flat;
    FPinterp *pyz_flat;
    FPinterp *pzz_flat;
    
};

struct GPUInterpDensSpecies* gpuInterpDensSpeciesAllocateAndCpyStatic(const struct grid& grid, const struct interpDensSpecies& interp_dens_species);

/**
 * @brief Allocate and copy dynamic data of the struct interpDensSpecies to the GPU
 * @param grid: grid information
 * @param interp_dens_species: interpolated densities of species
 * @param 
 */
void gpuInterpDensSpeciesCpyTo(const struct grid& grid, const struct interpDensSpecies& interp_dens_species, struct GPUInterpDensSpecies* gpu_interp_dens_species);

/**
 * @brief Copy back dynamic data of the struct interpDensSpecies from the GPU to the CPU
 * @param grid: grid information
 * @param interp_dens_species: interpolated densities of species
 * @param : GPU struct
 */
void gpuInterpDensSpeciesCpyBack(const struct grid& grid, struct interpDensSpecies& interp_dens_species, const struct GPUInterpDensSpecies* gpu_interp_dens_species);

/**
 * @brief Deallocate the dynamic data of the struct GPUInterpDensSpecies
 * @param 
 */
void gpuInterpDensSpeciesDeallocate(struct GPUInterpDensSpecies*);


#endif