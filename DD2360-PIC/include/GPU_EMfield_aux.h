#ifndef GPU_EMFIELD_AUX_H
#define GPU_EMFIELD_AUX_H

#include "Alloc.h"
#include "Grid.h"
#include "cudaAux.h"
#include "EMfield_aux.h"

/** structure with auxiliary field quantities like potentials or quantities defined at centers  */
struct GPUEMfield_aux {
    
    
    /* Electrostatic potential defined on central points*/
    FPfield* Phi_flat;

    /* Electric field at time theta */
    FPfield* Exth_flat;
    FPfield* Eyth_flat;
    FPfield* Ezth_flat;

    /* Magnetic field defined on nodes: last index is component - Centers */
    FPfield* Bxc_flat;
    FPfield* Byc_flat;
    FPfield* Bzc_flat;
};

/**
 * @brief: allocate electric and magnetic field
 * @param: grd -> grid with size information
 * @param: gpu_emf_aux -> the auxiliary field quantities to be alloc 
 * @param: emf_aux -> the auxiliary field quantities to be copied
 */
void gpuFieldAuxAllocateAndCpy(const struct grid&, struct GPUEMfield_aux*, const struct EMfield_aux&);

/**
 * @brief: deallocate electric and magnetic field
 * @param: gpu_emf_aux -> the auxiliary field quantities to be dealloc
 */
void gpuFieldAuxDeallocate(struct GPUEMfield_aux*);



#endif