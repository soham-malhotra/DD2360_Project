#ifndef GPU_PARTICLES_H
#define GPU_PARTICLES_H

#include "Alloc.h"
#include "Particles.h"
#include "cudaAux.h"

// using the original struct for now
struct GPUParticles : particles{};

void gpuParticleAllocateAndCpy(const struct grid&, struct GPUParticles*, const struct particles&);

void gpuParticleDeallocate(struct GPUParticles*);

#endif