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

struct GPUInterpDensSpecies* gpuInterpDensSpeciesAllocateAndCpy(const struct grid&, const struct interpDensSpecies&);

void gpuInterpDensSpeciesDeallocate(struct GPUInterpDensSpecies*);


#endif