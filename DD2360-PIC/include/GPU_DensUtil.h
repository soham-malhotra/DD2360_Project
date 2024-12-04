#include "GPU_Grid.h"
#include "GPU_InterpDensSpecies.h"
#include "GPU_InterpDensNet.h"

void gpu_setZeroDensities(struct GPUInterpDensSpecies** gpu_ids, struct GPUInterpDensNet* gpu_idn, struct GPUGrid* gpu_grd, struct parameters* param, struct grid* grd);

void gpu_applyBCids(struct GPUInterpDensSpecies** gpu_ids,  struct GPUGrid* gpu_grd, parameters* param, struct grid* grd);

template <class IDTYPE>
__global__ void setZeroDensitiesNode_kernel(IDTYPE* id, struct GPUGrid* gpu_grd);

template <class IDTYPE>
__global__ void setZeroDensitiesCell_kernel(IDTYPE* id, struct GPUGrid* gpu_grd);

__global__ void applyBCids_kernel(struct GPUInterpDensSpecies* ids, struct GPUGrid* gpu_grd, __grid_constant__ const struct parameters param);