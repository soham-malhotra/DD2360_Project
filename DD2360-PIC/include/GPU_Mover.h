#include "GPU_Parameters.h"
#include "GPU_Grid.h"
#include "GPU_EMfield.h"
#include "GPU_Particles.h"

void gpu_mover_PC(struct GPUParticles** gpu_part, struct GPUEMfield* gpu_field, struct GPUGrid* gpu_grd, struct particles** part, struct parameters* param);

__global__ void mover_PC_kernel(struct GPUParticles* gpu_part, struct GPUEMfield* gpu_field, struct GPUGrid* gpu_grd, __grid_constant__ const struct parameters param);