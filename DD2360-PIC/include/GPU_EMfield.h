#ifndef GPU_EMFIELD_H
#define GPU_EMFIELD_H

#include "Alloc.h"
#include "Grid.h"
#include "EMfield.h"
#include "cudaAux.h"

/** structure with field information */
struct GPUEMfield {
    // field arrays
    
    /* Electric field defined on nodes: last index is component */
    FPfield* Ex_flat;
    FPfield* Ey_flat;
    FPfield* Ez_flat;

    /* Magnetic field defined on nodes: last index is component */
    FPfield* Bxn_flat;
    FPfield* Byn_flat;
    FPfield* Bzn_flat;   
};

/**
 * @brief: allocate electric and magnetic field on GPU
 * @param: grid -> the grid with the size param
 * @param: GPUEMfield -> the field to allocate and copy to
 * @param: EMfield -> the field to copy
 */
struct GPUEMfield* gpuFieldAllocateAndCpy(const struct grid&, const struct EMfield&);

/**
 * @brief: deallocate electric and magnetic field on GPU
 * @param: GPUEMfield -> the field to deallocate
 */
void gpuFieldDeallocate(struct GPUEMfield*);


#endif