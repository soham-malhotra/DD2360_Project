#include "GPU_Particles.h"

struct GPUParticles* gpuParticleAllocateAndCpyStatic(const struct particles& particles) {
    GPUParticles* gpu_particles = nullptr;

    size_t size_arr = particles.npmax * sizeof(FPpart);  // size of particle position and velocity arrays

    cudaErrorHandling(cudaMalloc(&gpu_particles, sizeof(GPUParticles)));

    // copy static members
    cudaErrorHandling(cudaMemcpy(&(gpu_particles->species_ID), &particles.species_ID, sizeof(particles.species_ID), cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMemcpy(&(gpu_particles->npmax), &particles.npmax, sizeof(particles.npmax), cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMemcpy(&(gpu_particles->nop), &particles.nop, sizeof(particles.nop), cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMemcpy(&(gpu_particles->NiterMover), &particles.NiterMover, sizeof(particles.NiterMover), cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMemcpy(&(gpu_particles->n_sub_cycles), &particles.n_sub_cycles, sizeof(particles.n_sub_cycles), cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMemcpy(&(gpu_particles->npcel), &particles.npcel, sizeof(particles.npcel), cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMemcpy(&(gpu_particles->npcelx), &particles.npcelx, sizeof(particles.npcelx), cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMemcpy(&(gpu_particles->npcely), &particles.npcely, sizeof(particles.npcely), cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMemcpy(&(gpu_particles->npcelz), &particles.npcelz, sizeof(particles.npcelz), cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMemcpy(&(gpu_particles->qom), &particles.qom, sizeof(particles.qom), cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMemcpy(&(gpu_particles->u0), &particles.u0, sizeof(particles.u0), cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMemcpy(&(gpu_particles->v0), &particles.v0, sizeof(particles.v0), cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMemcpy(&(gpu_particles->w0), &particles.w0, sizeof(particles.w0), cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMemcpy(&(gpu_particles->uth), &particles.uth, sizeof(particles.uth), cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMemcpy(&(gpu_particles->vth), &particles.vth, sizeof(particles.vth), cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMemcpy(&(gpu_particles->wth), &particles.wth, sizeof(particles.wth), cudaMemcpyHostToDevice));
    cudaErrorHandling(cudaMemcpy(&(gpu_particles->q), &particles.q, sizeof(particles.q), cudaMemcpyHostToDevice));

    // allocate positions
    allocateDeviceArray<FPpart>(&gpu_particles->x, size_arr);
    allocateDeviceArray<FPpart>(&gpu_particles->y, size_arr);
    allocateDeviceArray<FPpart>(&gpu_particles->z, size_arr);

    // allocate velocities
    allocateDeviceArray<FPpart>(&gpu_particles->u, size_arr);
    allocateDeviceArray<FPpart>(&gpu_particles->v, size_arr);
    allocateDeviceArray<FPpart>(&gpu_particles->w, size_arr);

    // allocate charges (+ statistical weights)
    allocateAndCpyDeviceArray<FPpart>(&gpu_particles->q, particles.q, size_arr);

    return gpu_particles;
}

void gpuParticleCpyTo(const struct particles& particles, struct GPUParticles* gpu_particles) {
    GPUParticles temp_particles;
    cudaErrorHandling(cudaMemcpy(&temp_particles, gpu_particles, sizeof(GPUParticles), cudaMemcpyDeviceToHost));

    size_t size_arr = particles.npmax * sizeof(FPpart);  // size of particle position and velocity arrays

    // copy positions
    copyArrayToDevice<FPpart>(temp_particles.x, particles.x, size_arr);
    copyArrayToDevice<FPpart>(temp_particles.y, particles.y, size_arr);
    copyArrayToDevice<FPpart>(temp_particles.z, particles.z, size_arr);

    // copy velocities
    copyArrayToDevice<FPpart>(temp_particles.u, particles.u, size_arr);
    copyArrayToDevice<FPpart>(temp_particles.v, particles.v, size_arr);
    copyArrayToDevice<FPpart>(temp_particles.w, particles.w, size_arr);
}

void gpuParticleCpyBack(struct particles& particles, const struct GPUParticles* gpu_particles) {
    GPUParticles temp_particles;
    cudaErrorHandling(cudaMemcpy(&temp_particles, gpu_particles, sizeof(GPUParticles), cudaMemcpyDeviceToHost));

    size_t size_arr = particles.npmax * sizeof(FPpart);  // size of particle position and velocity arrays

    // copy positions
    copyArrayFromDevice<FPpart>(particles.x, temp_particles.x, size_arr);
    copyArrayFromDevice<FPpart>(particles.y, temp_particles.y, size_arr);
    copyArrayFromDevice<FPpart>(particles.z, temp_particles.z, size_arr);

    // copy velocities
    copyArrayFromDevice<FPpart>(particles.u, temp_particles.u, size_arr);
    copyArrayFromDevice<FPpart>(particles.v, temp_particles.v, size_arr);
    copyArrayFromDevice<FPpart>(particles.w, temp_particles.w, size_arr);
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

    //deallocate charges
    cudaErrorHandling(cudaFree(temp_particles.q));

    //deallocate the struct itself
    cudaErrorHandling(cudaFree(gpu_particles));
}