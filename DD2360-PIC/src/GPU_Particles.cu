#include "GPU_Particles.h"

void gpuParticleAllocateAndCpy(const struct grid& grid, struct GPUParticles* gpu_particles, const struct particles& particles) {

    size_t size_arr = particles.npmax * sizeof(FPpart);  // size of particle position and velocity arrays

    cudaErrorHandling(cudaMalloc(&gpu_particles, sizeof(particles)));
    cudaErrorHandling(cudaMemcpy(gpu_particles, &particles, sizeof(particles), cudaMemcpyHostToDevice));

    // allocate positions
    copyArrayToDeviceStruct<FPpart>(&(gpu_particles->x), particles.x, size_arr);
    copyArrayToDeviceStruct<FPpart>(&(gpu_particles->y), particles.y, size_arr);
    copyArrayToDeviceStruct<FPpart>(&(gpu_particles->z), particles.z, size_arr);

    // allocate velocities
    copyArrayToDeviceStruct<FPpart>(&(gpu_particles->u), particles.u, size_arr);
    copyArrayToDeviceStruct<FPpart>(&(gpu_particles->v), particles.v, size_arr);
    copyArrayToDeviceStruct<FPpart>(&(gpu_particles->w), particles.w, size_arr);
}

void gpuParticleDeallocate(struct GPUParticles* gpu_particles) {
    //deallocate positions
    cudaErrorHandling(cudaFree(gpu_particles->x));
    cudaErrorHandling(cudaFree(gpu_particles->y));
    cudaErrorHandling(cudaFree(gpu_particles->z));

    //deallocate velocities
    cudaErrorHandling(cudaFree(gpu_particles->u));
    cudaErrorHandling(cudaFree(gpu_particles->v));
    cudaErrorHandling(cudaFree(gpu_particles->w));

    //deallocate the struct itself
    cudaErrorHandling(cudaFree(gpu_particles));
}