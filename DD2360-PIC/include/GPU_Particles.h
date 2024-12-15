#ifndef GPU_PARTICLES_H
#define GPU_PARTICLES_H

#include "Alloc.h"
#include "Particles.h"
#include "cudaAux.h"

// using the original struct for now
struct GPUParticles : particles{};

struct GPUParticles* gpuParticleAllocateAndCpyStatic(const struct particles& particles);

void gpuParticleCpyTo(const struct particles& particles, struct GPUParticles* gpu_particles);

void gpuParticleCpyBack(struct particles& particles, const struct GPUParticles* gpu_particles);

void gpuParticleDeallocate(struct GPUParticles*);

#endif