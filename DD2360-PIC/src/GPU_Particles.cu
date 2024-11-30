#include "GPU_Particles.h"

struct GPUParticles* gpuParticleAllocateAndCpy(const struct grid& grid, const struct particles& particles) {
    GPUParticles* gpu_particles = nullptr;

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

    return gpu_particles;
}

void gpuParticleDeallocate(struct GPUParticles* gpu_particles) {
    GPUParticles temp_particles;
    cudaErrorHandling(cudaMemcpy(&temp_particles, gpu_particles, sizeof(GPUParticles), cudaMemcpyDeviceToHost));

    //deallocate positions
    cudaErrorHandling(cudaFree(temp_particles.x));
    cudaErrorHandling(cudaFree(temp_particles.y));
    cudaErrorHandling(cudaFree(temp_particles.z));

    //deallocate velocities
    cudaErrorHandling(cudaFree(temp_particles.u));
    cudaErrorHandling(cudaFree(temp_particles.v));
    cudaErrorHandling(cudaFree(temp_particles.w));

    //deallocate the struct itself
    cudaErrorHandling(cudaFree(gpu_particles));
}