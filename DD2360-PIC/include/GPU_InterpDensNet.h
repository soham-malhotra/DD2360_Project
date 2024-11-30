#ifndef GPU_INTERPDENSNET_H
#define GPU_INTERPDENSNET_H

#include "Alloc.h"
#include "Grid.h"
#include "InterpDensNet.h"
#include "cudaAux.h"

struct GPUInterpDensNet {
    
    /** charged densities */
    FPinterp *rhon_flat; // rho defined on nodes
    FPinterp *rhoc_flat; // rho defined at center cell
    /** J current densities */
    FPinterp *Jx_flat;
    FPinterp *Jy_flat;
    FPinterp *Jz_flat; // on nodes
    /** p = pressure tensor*/
    FPinterp *pxx_flat;
    FPinterp *pxy_flat;
    FPinterp *pxz_flat; // on nodes
    FPinterp *pyy_flat;
    FPinterp *pyz_flat;
    FPinterp *pzz_flat; // on nodes
};

struct GPUInterpDensNet* gpuInterpDensNetAllocateAndCpy(const struct grid&, const struct interpDensNet&);

void gpuInterpDensNetDeallocate(struct GPUInterpDensNet*);


#endif