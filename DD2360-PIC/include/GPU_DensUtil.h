#include "GPU_Grid.h"
#include "GPU_InterpDensSpecies.h"
#include "GPU_InterpDensNet.h"

/**
 * @brief Set the densities of all species to zero on the grid nodes.
 * @param gpu_ids Array of pointers to the species data on the GPU.
 * @param gpu_idn Pointer to the net density data on the GPU.
 * @param gpu_grd Pointer to the grid data on the GPU.
 * @param param Pointer to the parameters struct.
 * @param grd Pointer to the grid struct.
 */
void gpu_setZeroDensities(struct GPUInterpDensSpecies** gpu_ids, struct GPUInterpDensNet* gpu_idn, struct GPUGrid* gpu_grd, struct parameters* param, struct grid* grd);

/**
 * @brief Set the densities of all species to zero on the grid nodes.
 * @param gpu_ids Array of pointers to the species data on the GPU.
 * @param gpu_idn Pointer to the net density data on the GPU.
 * @param gpu_grd Pointer to the grid data on the GPU.
 * @param param Pointer to the parameters struct.
 * @param grd Pointer to the grid struct.
 */
void gpu_applyBCids(struct GPUInterpDensSpecies** gpu_ids,  struct GPUGrid* gpu_grd, parameters* param, struct grid* grd);

/**
 * @brief Sum the densities of all species on the grid nodes.
 * @param gpu_ids Array of pointers to the species data on the GPU.
 * @param gpu_idn Pointer to the net density data on the GPU.
 * @param gpu_grd Pointer to the grid data on the GPU.
 * @param param Pointer to the parameters struct.
 * @param grd Pointer to the grid struct.
 */
void gpu_sumOverSpecies(struct GPUInterpDensNet* gpu_idn, struct GPUInterpDensSpecies** gpu_ids, struct GPUGrid* gpu_grd, struct parameters* param, struct grid* grd);

/**
 * @brief Apply boundary conditions to the scalar density on the grid nodes.
 * @param gpu_idn Pointer to the net density data on the GPU.
 * @param gpu_grd Pointer to the grid data on the GPU.
 * @param param Pointer to the parameters struct.
 */
void gpu_applyBCscalarDensN(struct GPUInterpDensNet* gpu_idn, struct GPUGrid* gpu_grd, struct grid* grd, struct parameters* param);

/**
 * @brief Set the densities of all species to zero on the grid nodes.
 * @param id Array of species IDs.
 * @param gpu_grd Pointer to the grid data on the GPU.
 */
template <class IDTYPE>
__global__ void setZeroDensitiesNode_kernel(IDTYPE* id, struct GPUGrid* gpu_grd);

/**
 * @brief Set the densities of all species to zero on the grid cells.
 * @param id Array of species IDs.
 * @param gpu_grd Pointer to the grid data on the GPU.
 */
template <class IDTYPE>
__global__ void setZeroDensitiesCell_kernel(IDTYPE* id, struct GPUGrid* gpu_grd);

/// Apply boundary conditions to the densities of all species on the grid nodes
__global__ void applyBCids_periodicX(struct GPUInterpDensSpecies* ids, struct GPUGrid* gpu_grd, __grid_constant__ const struct parameters param);

__global__ void applyBCids_periodicY(struct GPUInterpDensSpecies* ids, struct GPUGrid* gpu_grd, __grid_constant__ const struct parameters param);

__global__ void applyBCids_periodicZ(struct GPUInterpDensSpecies* ids, struct GPUGrid* gpu_grd, __grid_constant__ const struct parameters param);

__global__ void applyBCids_nonPeriodicX(struct GPUInterpDensSpecies* ids, struct GPUGrid* gpu_grd, __grid_constant__ const struct parameters param);

__global__ void applyBCids_nonPeriodicY(struct GPUInterpDensSpecies* ids, struct GPUGrid* gpu_grd, __grid_constant__ const struct parameters param);

__global__ void applyBCids_nonPeriodicZ(struct GPUInterpDensSpecies* ids, struct GPUGrid* gpu_grd, __grid_constant__ const struct parameters param);

__global__ void applyBCscalarDensN_periodicX(struct GPUInterpDensNet* idn, struct GPUGrid* gpu_grd, __grid_constant__ const struct parameters param);

__global__ void applyBCscalarDensN_periodicY(struct GPUInterpDensNet* idn, struct GPUGrid* gpu_grd, __grid_constant__ const struct parameters param);

__global__ void applyBCscalarDensN_periodicZ(struct GPUInterpDensNet* idn, struct GPUGrid* gpu_grd, __grid_constant__ const struct parameters param);

__global__ void applyBCscalarDensN_edgeX(struct GPUInterpDensNet* idn, struct GPUGrid* gpu_grd, __grid_constant__ const struct parameters param);

__global__ void applyBCscalarDensN_edgeY(struct GPUInterpDensNet* idn, struct GPUGrid* gpu_grd, __grid_constant__ const struct parameters param);

__global__ void applyBCscalarDensN_edgeZ(struct GPUInterpDensNet* idn, struct GPUGrid* gpu_grd, __grid_constant__ const struct parameters param);

__global__ void applyBCscalarDensN_corners(struct GPUInterpDensNet* idn, struct GPUGrid* gpu_grd, __grid_constant__ const struct parameters param);

__global__ void applyBCscalarDensN_nonPeriodicX(struct GPUInterpDensNet* idn, struct GPUGrid* gpu_grd, __grid_constant__ const struct parameters param);

__global__ void applyBCscalarDensN_nonPeriodicY(struct GPUInterpDensNet* idn, struct GPUGrid* gpu_grd, __grid_constant__ const struct parameters param);

__global__ void applyBCscalarDensN_nonPeriodicZ(struct GPUInterpDensNet* idn, struct GPUGrid* gpu_grd, __grid_constant__ const struct parameters param);

__global__ void sumOverSpeciesNode_kernel(struct GPUInterpDensNet* idn, struct GPUInterpDensSpecies* ids, struct GPUGrid* gpu_grd);
