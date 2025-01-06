#include "GPU_Parameters.h"
#include "GPU_Grid.h"
#include "GPU_InterpDensSpecies.h"
#include "GPU_Particles.h"

// void gpu_interpP2G(struct GPUParticles** gpu_part, struct GPUInterpDensSpecies** gpu_ids, struct GPUGrid* gpu_grd, struct particles** part, struct parameters* param);
/**
 * @brief Interpolates particle quantities to the grid
 */
void gpu_interpP2G(struct GPUParticles** gpu_part, struct GPUInterpDensSpecies** gpu_ids, struct GPUGrid* gpu_grd, struct particles** part, struct grid* grd, struct parameters* param);

/**
 * @brief Interpolates particle quantities to the grid, in kernel form 
 */
__global__ void interpP2G_kernel(struct GPUParticles* gpu_part, struct GPUInterpDensSpecies* gpu_ids, struct GPUGrid* gpu_grd);
