#include "GPU_Particles.h"

void gpu_sort_particles(struct GPUParticles** gpu_part, struct particles* part, struct parameters* param, struct grid* grd);

__global__ void categorize_kernel(struct GPUParticles* gpu_part);