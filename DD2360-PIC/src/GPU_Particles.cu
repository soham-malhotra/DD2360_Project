#include "GPU_Particles.h"

void gpuParticleAllocateAndCpy(const struct grid& grid, struct GPUParticles* gpu_particles, const struct particles& particles) {

    size_t size_arr = particles.npmax * sizeof(FPpart);  // size of particle position and velocity arrays

    cudaErrorHandling(cudaMalloc(&gpu_particles, sizeof(particles)));
    cudaErrorHandling(cudaMemcpy(gpu_particles, &particles, sizeof(particles), cudaMemcpyHostToDevice));

    // allocate positions
    cudaErrorHandling(cudaMalloc(&gpu_particles->x, size_arr));
    cudaErrorHandling(cudaMemcpy(gpu_particles->x, particles.x, size_arr, cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMalloc(&gpu_particles->y, size_arr));
    cudaErrorHandling(cudaMemcpy(gpu_particles->y, particles.y, size_arr, cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMalloc(&gpu_particles->z, size_arr));
    cudaErrorHandling(cudaMemcpy(gpu_particles->z, particles.z, size_arr, cudaMemcpyHostToDevice));

    // allocate velocities
    cudaErrorHandling(cudaMalloc(&gpu_particles->u, size_arr));
    cudaErrorHandling(cudaMemcpy(gpu_particles->u, particles.u, size_arr, cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMalloc(&gpu_particles->v, size_arr));
    cudaErrorHandling(cudaMemcpy(gpu_particles->v, particles.v, size_arr, cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMalloc(&gpu_particles->w, size_arr));
    cudaErrorHandling(cudaMemcpy(gpu_particles->w, particles.w, size_arr, cudaMemcpyHostToDevice));
}