#include "GPU_Particles.h"

/**
 * @brief: sorts particles into cells
 * @param: gpu_part -> the particles to sort
 * @param: part -> the particles
 * @param: param -> the parameters
 * @param: grd -> the grid
 */
void gpu_sort_particles(struct GPUParticles** gpu_part, struct particles* part, struct parameters* param, struct grid* grd);

/**
 * @brief: categorizes particles into cells
 * @param: gpu_part -> the particles to categorize
 */
__global__ void categorize_kernel(struct GPUParticles* gpu_part);