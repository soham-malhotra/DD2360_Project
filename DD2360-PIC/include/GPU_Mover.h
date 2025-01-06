#include "GPU_Parameters.h"
#include "GPU_Grid.h"
#include "GPU_EMfield.h"
#include "GPU_Particles.h"

/**
 * @brief: moves particles on GPU
 * @param: gpu_part -> the particles to move
 * @param: gpu_field -> the field to move the particles in
 * @param: gpu_grd -> the grid to move the particles in
 * @param: part -> the particles to move
 * @param: param -> the parameters
 */
void gpu_mover_PC(struct GPUParticles** gpu_part, struct GPUEMfield* gpu_field, struct GPUGrid* gpu_grd, struct particles** part, struct parameters* param);

/**
 * @brief: moves particles on GPU
 * @param: gpu_part -> the particles to move
 * @param: gpu_field -> the field to move the particles in
 * @param: gpu_grd -> the grid to move the particles in
 * @param: param -> the parameters
 * @param: qomdt2 -> qom*dt/2 (species specific)
 * @param: dt_sub_cycling -> dt/subcycling (species specific)
 * @param: dto2 -> dt/2 (species specific)
 */
__global__ void mover_PC_kernel(struct GPUParticles* gpu_part, struct GPUEMfield* gpu_field, 
struct GPUGrid* gpu_grd, __grid_constant__ const struct parameters param,
const FPpart qomdt2, const FPpart dt_sub_cycling, const FPpart dto2);