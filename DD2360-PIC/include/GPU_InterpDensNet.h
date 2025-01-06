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

/**
 * @brief: allocate and copy static data to GPU
 * @param: grid -> the grid to allocate for
 */
struct GPUInterpDensNet* gpuInterpDensNetAllocate(const struct grid& grid);

/**
 * @brief: copy data to GPU
 * @param: grid -> the grid to copy for
 * @param: interp_dens_net -> the data to copy
 * @param: gpu_interp_dens_net -> the GPU data to copy to
 */
void gpuInterpDensNetCpyTo(const struct grid& grid, const struct interpDensNet& interp_dens_net, struct GPUInterpDensNet* gpu_interp_dens_net);

/**
 * @brief: copy data back to CPU
 * @param: grid -> the grid to copy for
 * @param: interp_dens_net -> the data to copy back
 * @param: gpu_interp_dens_net -> the GPU data to copy from
 */
void gpuInterpDensNetCpyBack(const struct grid& grid, struct interpDensNet& interp_dens_net, const struct GPUInterpDensNet* gpu_interp_dens_net);

/**
 * @brief: deallocate GPU data
 * @param: gpu_interp_dens_net -> the GPU data to deallocate
 */
void gpuInterpDensNetDeallocate(struct GPUInterpDensNet*);


#endif