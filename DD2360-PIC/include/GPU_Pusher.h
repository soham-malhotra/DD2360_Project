#include "GPU_Parameters.h"
#include "GPU_Grid.h"
#include "GPU_EMfield.h"
#include "GPU_Particles.h"

void particle_subcycler(struct GPUParticles* gpu_part, struct GPUEMfield* gpu_field, struct GPUGrid* gpu_grd, struct GPUParameters* gpu_param, 
                        struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param);

__global__ void particle_push_kernel(struct GPUParticles* gpu_part, struct GPUEMfield* gpu_field, struct GPUGrid* gpu_grd, struct GPUParameters* gpu_param);