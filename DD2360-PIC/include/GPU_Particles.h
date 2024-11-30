#ifndef GPU_PARTICLES_H
#define GPU_PARTICLES_H

#include "Alloc.h"
#include "Particles.h"
#include "cudaAux.h"

// using the original struct for now
struct GPUParticles : particles{};

struct GPUParticles* gpuParticleAllocateAndCpy(const struct grid&, const struct particles&);

void gpuParticleDeallocate(struct GPUParticles*);

#endif