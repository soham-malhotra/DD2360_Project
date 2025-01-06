#ifndef GPU_PARTICLES_H
#define GPU_PARTICLES_H

#define MAX_PART_PER_CELL 200

#include "Alloc.h"
#include "Particles.h"
#include "cudaAux.h"

// using the original struct for now
struct GPUParticles {
    /** species ID: 0, 1, 2 , ... */
    int species_ID;
    
    /** maximum number of particles of this species on this domain. used for memory allocation */
    long npmax;
    /** number of particles of this species on this domain */
    long nop;
    
    /** Electron and ions have different number of iterations: ions moves slower than ions */
    int NiterMover;
    /** number of particle of subcycles in the mover */
    int n_sub_cycles;
    
    
    /** number of particles per cell */
    int npcel;
    /** number of particles per cell - X direction */
    int npcelx;
    /** number of particles per cell - Y direction */
    int npcely;
    /** number of particles per cell - Z direction */
    int npcelz;
    
    
    /** charge over mass ratio */
    FPpart qom;
    
    /* drift and thermal velocities for this species */
    FPpart u0, v0, w0;
    FPpart uth, vth, wth;
    
    /** particle arrays: 1D arrays[npmax] */
    FPpart* x; FPpart*  y; FPpart* z; FPpart* u; FPpart* v; FPpart* w;
    /** q must have precision of interpolated quantities: typically double. Not used in mover */
    FPinterp* q;

    int* cell_id;  // stored which cell each particle is in, has length nop
    int* cell_counter;  // stored how many particles are in each cell, has length grd.nxc * grd.nyc * grd.nzc
    FPpart *cell_x, *cell_y, *cell_z, *cell_u, *cell_v, *cell_w;  // 2d arrays, each row is a cell, each column is a particle in that cell
    FPinterp *cell_q;  // contiguous blocks! (boundaries not enforced)
};


struct GPUParticles* gpuParticleAllocateAndCpyStatic(const struct grid& grd, const struct particles& particles);

/**
 * @brief: ports particles to GPU
 * @param: particles -> the particles to port
 */
void gpuParticleCpyTo(const struct particles& particles, struct GPUParticles* gpu_particles);

/**
 * @brief: ports particles back to CPU
 * @param: particles -> the particles to port back
 */
void gpuParticleCpyBack(struct particles& particles, const struct GPUParticles* gpu_particles);

/**
 * @brief: deallocates particles on GPU
 * @param: particles -> the particles to deallocate
 */
void gpuParticleDeallocate(struct GPUParticles*);

#endif