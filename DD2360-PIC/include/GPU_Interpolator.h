#include "GPU_Parameters.h"
#include "GPU_Grid.h"
#include "GPU_InterpDensSpecies.h"
#include "GPU_Particles.h"

void gpu_interpP2G(struct GPUParticles** gpu_part, struct GPUInterpDensSpecies** gpu_ids, struct GPUGrid* gpu_grd, struct particles** part, struct parameters* param);

__global__ void interpP2G_kernel(struct GPUParticles* gpu_part, struct GPUInterpDensSpecies* gpu_ids, struct GPUGrid* gpu_grd);