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

struct GPUInterpDensNet* gpuInterpDensNetAllocate(const struct grid& grid);

void gpuInterpDensNetCpyTo(const struct grid& grid, const struct interpDensNet& interp_dens_net, struct GPUInterpDensNet* gpu_interp_dens_net);

void gpuInterpDensNetCpyBack(const struct grid& grid, struct interpDensNet& interp_dens_net, const struct GPUInterpDensNet* gpu_interp_dens_net);

void gpuInterpDensNetDeallocate(struct GPUInterpDensNet*);


#endif