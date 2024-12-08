#include "GPU_DensUtil.h"

#define THREAD_NR_PER_X 16.0
#define THREAD_NR_PER_Y 16.0
#define THREAD_NR_PER_Z 1.0

void gpu_setZeroDensities(struct GPUInterpDensSpecies** gpu_ids, struct GPUInterpDensNet* gpu_idn, struct GPUGrid* gpu_grd, struct parameters* param, struct grid* grd) {

    dim3 dimBlock(THREAD_NR_PER_X, THREAD_NR_PER_Y, THREAD_NR_PER_Z);
    dim3 dimGridNode(ceil(grd->nxn/THREAD_NR_PER_X), ceil(grd->nyn/THREAD_NR_PER_Y), ceil(grd->nzn/THREAD_NR_PER_Z));
    dim3 dimGridCell(ceil(grd->nxc/THREAD_NR_PER_X), ceil(grd->nyc/THREAD_NR_PER_Y), ceil(grd->nzc/THREAD_NR_PER_Z));

    for (int is = 0; is < param->ns; is++) {  // lot of overhead for calling kernels for something trivial
        setZeroDensitiesNode_kernel<GPUInterpDensSpecies><<<dimGridNode, dimBlock>>>(gpu_ids[is], gpu_grd);
        cudaDeviceSynchronize();  // not really necessary?
        setZeroDensitiesCell_kernel<GPUInterpDensSpecies><<<dimGridCell, dimBlock>>>(gpu_ids[is], gpu_grd);
        cudaDeviceSynchronize();
    }

    setZeroDensitiesNode_kernel<GPUInterpDensNet><<<dimGridNode, dimBlock>>>(gpu_idn, gpu_grd);
    cudaDeviceSynchronize();
    setZeroDensitiesCell_kernel<GPUInterpDensNet><<<dimGridCell, dimBlock>>>(gpu_idn, gpu_grd);
    cudaDeviceSynchronize();
}

// boundary conditions for actual nodes
void gpu_applyBCids(struct GPUInterpDensSpecies** gpu_ids,  struct GPUGrid* gpu_grd, parameters* param, struct grid* grd) {
    // The same corner points are modified by differenc BCs. This has to be done in the exact order as the original code.
    // As entire grids cannot be synced from within kernel code, for the current approach, 6 different kernel calls are needed...
    // TODO Still might not be exact for some reason!

    dim3 dimBlock(THREAD_NR_PER_X, THREAD_NR_PER_Y, THREAD_NR_PER_Z);
    dim3 dimGrid(ceil(grd->nxn/THREAD_NR_PER_X), ceil(grd->nyn/THREAD_NR_PER_Y), ceil(grd->nzn/THREAD_NR_PER_Z));

    for (int is = 0; is < param->ns; is++) {
        // applyBCids_kernel<<<dimGrid, dimBlock>>>(gpu_ids[is], gpu_grd, *param);
        applyBCids_periodicX<<<dimGrid, dimBlock>>>(gpu_ids[is], gpu_grd, *param);
        cudaDeviceSynchronize();
        applyBCids_periodicY<<<dimGrid, dimBlock>>>(gpu_ids[is], gpu_grd, *param);
        cudaDeviceSynchronize();
        applyBCids_periodicZ<<<dimGrid, dimBlock>>>(gpu_ids[is], gpu_grd, *param);
        cudaDeviceSynchronize();
        applyBCids_nonPeriodicX<<<dimGrid, dimBlock>>>(gpu_ids[is], gpu_grd, *param);
        cudaDeviceSynchronize();
        applyBCids_nonPeriodicY<<<dimGrid, dimBlock>>>(gpu_ids[is], gpu_grd, *param);
        cudaDeviceSynchronize();
        applyBCids_nonPeriodicZ<<<dimGrid, dimBlock>>>(gpu_ids[is], gpu_grd, *param);
        cudaDeviceSynchronize();
    }
}

// boundary conditions for ghost nodes
void gpu_applyBCscalarDensN(struct GPUInterpDensNet* gpu_idn, struct GPUGrid* gpu_grd, struct grid* grd) {

    dim3 dimBlock(THREAD_NR_PER_X, THREAD_NR_PER_Y, THREAD_NR_PER_Z);
    dim3 dimGrid(ceil(grd->nxn/THREAD_NR_PER_X), ceil(grd->nyn/THREAD_NR_PER_Y), ceil(grd->nzn/THREAD_NR_PER_Z));

}

void gpu_sumOverSpecies(struct GPUInterpDensNet* gpu_idn, struct GPUInterpDensSpecies** gpu_ids, struct GPUGrid* gpu_grd, struct parameters* param, struct grid* grd) {

    dim3 dimBlock(THREAD_NR_PER_X, THREAD_NR_PER_Y, THREAD_NR_PER_Z);
    dim3 dimGridNode(ceil(grd->nxn/THREAD_NR_PER_X), ceil(grd->nyn/THREAD_NR_PER_Y), ceil(grd->nzn/THREAD_NR_PER_Z));
    dim3 dimGridCell(ceil(grd->nxc/THREAD_NR_PER_X), ceil(grd->nyc/THREAD_NR_PER_Y), ceil(grd->nzc/THREAD_NR_PER_Z));

    for (int is = 0; is < param->ns; is++) {  // this is launching a kernel that performs a single addition for each quantity
        sumOverSpeciesNode_kernel<<<dimGridNode, dimBlock>>>(gpu_idn, gpu_ids[is], gpu_grd);
        cudaDeviceSynchronize();
    }
}

template <class IDTYPE>
__global__ void setZeroDensitiesNode_kernel(IDTYPE* id, struct GPUGrid* gpu_grd) {
    
        int ix = blockIdx.x * blockDim.x + threadIdx.x;
        int iy = blockIdx.y * blockDim.y + threadIdx.y;
        int iz = blockIdx.z * blockDim.z + threadIdx.z;
    
        if (ix < gpu_grd->nxn && iy < gpu_grd->nyn && iz < gpu_grd->nzn) {
            int ind = ix * gpu_grd->nyn * gpu_grd->nzn + iy * gpu_grd->nzn + iz;
            id->rhon_flat[ind] = 0.0;
            id->Jx_flat[ind] = 0.0;
            id->Jy_flat[ind] = 0.0;
            id->Jz_flat[ind] = 0.0;
        }
}

template <class IDTYPE>
__global__ void setZeroDensitiesCell_kernel(IDTYPE* id, struct GPUGrid* gpu_grd) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix < gpu_grd->nxc && iy < gpu_grd->nyc && iz < gpu_grd->nzc) {
        int ind = ix * gpu_grd->nyc * gpu_grd->nzc + iy * gpu_grd->nzc + iz;
        id->rhoc_flat[ind] = 0.0;
    }
}

__global__ void applyBCids_periodicX(struct GPUInterpDensSpecies* ids, struct GPUGrid* gpu_grd, __grid_constant__ const struct parameters param) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int iy = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int iz = blockIdx.z * blockDim.z + threadIdx.z + 1;

    int ny = gpu_grd->nyn;
    int nz = gpu_grd->nzn;
    int nx = gpu_grd->nxn;

    if (param.PERIODICX && ix == 1 && iy < ny - 1 && iz < nz - 1) {
        ids->rhon_flat[1 * ny * nz + iy * nz + iz] += ids->rhon_flat[(nx - 2) * ny * nz + iy * nz + iz];
        ids->rhon_flat[(nx - 2) * ny * nz + iy * nz + iz] = ids->rhon_flat[1 * ny * nz + iy * nz + iz];
        ids->Jx_flat[1 * ny * nz + iy * nz + iz] += ids->Jx_flat[(nx - 2) * ny * nz + iy * nz + iz];
        ids->Jx_flat[(nx - 2) * ny * nz + iy * nz + iz] = ids->Jx_flat[1 * ny * nz + iy * nz + iz];
        ids->Jy_flat[1 * ny * nz + iy * nz + iz] += ids->Jy_flat[(nx - 2) * ny * nz + iy * nz + iz];
        ids->Jy_flat[(nx - 2) * ny * nz + iy * nz + iz] = ids->Jy_flat[1 * ny * nz + iy * nz + iz];
        ids->Jz_flat[1 * ny * nz + iy * nz + iz] += ids->Jz_flat[(nx - 2) * ny * nz + iy * nz + iz];
        ids->Jz_flat[(nx - 2) * ny * nz + iy * nz + iz] = ids->Jz_flat[1 * ny * nz + iy * nz + iz];
    }
}

__global__ void applyBCids_periodicY(struct GPUInterpDensSpecies* ids, struct GPUGrid* gpu_grd, __grid_constant__ const struct parameters param) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int iy = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int iz = blockIdx.z * blockDim.z + threadIdx.z + 1;

    int ny = gpu_grd->nyn;
    int nz = gpu_grd->nzn;
    int nx = gpu_grd->nxn;

    if (param.PERIODICY && iy == 1 && ix < nx - 1 && iz < nz - 1) {
        ids->rhon_flat[ix * ny * nz + 1 * nz + iz] += ids->rhon_flat[ix * ny * nz + (ny - 2) * nz + iz];
        ids->rhon_flat[ix * ny * nz + (ny - 2) * nz + iz] = ids->rhon_flat[ix * ny * nz + 1 * nz + iz];
        ids->Jx_flat[ix * ny * nz + 1 * nz + iz] += ids->Jx_flat[ix * ny * nz + (ny - 2) * nz + iz];
        ids->Jx_flat[ix * ny * nz + (ny - 2) * nz + iz] = ids->Jx_flat[ix * ny * nz + 1 * nz + iz];
        ids->Jy_flat[ix * ny * nz + 1 * nz + iz] += ids->Jy_flat[ix * ny * nz + (ny - 2) * nz + iz];
        ids->Jy_flat[ix * ny * nz + (ny - 2) * nz + iz] = ids->Jy_flat[ix * ny * nz + 1 * nz + iz];
        ids->Jz_flat[ix * ny * nz + 1 * nz + iz] += ids->Jz_flat[ix * ny * nz + (ny - 2) * nz + iz];
        ids->Jz_flat[ix * ny * nz + (ny - 2) * nz + iz] = ids->Jz_flat[ix * ny * nz + 1 * nz + iz];
    }
}

__global__ void applyBCids_periodicZ(struct GPUInterpDensSpecies* ids, struct GPUGrid* gpu_grd, __grid_constant__ const struct parameters param) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int iy = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int iz = blockIdx.z * blockDim.z + threadIdx.z + 1;

    int ny = gpu_grd->nyn;
    int nz = gpu_grd->nzn;
    int nx = gpu_grd->nxn;

    if (param.PERIODICZ && iz == 1 && ix < nx - 1 && iy < ny - 1) {
        ids->rhon_flat[ix * ny * nz + iy * nz + 1] += ids->rhon_flat[ix * ny * nz + iy * nz + (nz - 2)];
        ids->rhon_flat[ix * ny * nz + iy * nz + (nz - 2)] = ids->rhon_flat[ix * ny * nz + iy * nz + 1];
        ids->Jx_flat[ix * ny * nz + iy * nz + 1] += ids->Jx_flat[ix * ny * nz + iy * nz + (nz - 2)];
        ids->Jx_flat[ix * ny * nz + iy * nz + (nz - 2)] = ids->Jx_flat[ix * ny * nz + iy * nz + 1];
        ids->Jy_flat[ix * ny * nz + iy * nz + 1] += ids->Jy_flat[ix * ny * nz + iy * nz + (nz - 2)];
        ids->Jy_flat[ix * ny * nz + iy * nz + (nz - 2)] = ids->Jy_flat[ix * ny * nz + iy * nz + 1];
        ids->Jz_flat[ix * ny * nz + iy * nz + 1] += ids->Jz_flat[ix * ny * nz + iy * nz + (nz - 2)];
        ids->Jz_flat[ix * ny * nz + iy * nz + (nz - 2)] = ids->Jz_flat[ix * ny * nz + iy * nz + 1];
    }
}

__global__ void applyBCids_nonPeriodicX(struct GPUInterpDensSpecies* ids, struct GPUGrid* gpu_grd, __grid_constant__ const struct parameters param) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int iy = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int iz = blockIdx.z * blockDim.z + threadIdx.z + 1;

    int ny = gpu_grd->nyn;
    int nz = gpu_grd->nzn;
    int nx = gpu_grd->nxn;

    if (!param.PERIODICX && ix == 1 && iy < ny - 1 && iz < nz - 1) {
        ids->rhon_flat[1 * ny * nz + iy * nz + iz] *= 2;
        ids->rhon_flat[(nx - 2) * ny * nz + iy * nz + iz] *= 2;
        ids->Jx_flat[1 * ny * nz + iy * nz + iz] *= 2;
        ids->Jx_flat[(nx - 2) * ny * nz + iy * nz + iz] *= 2;
        ids->Jy_flat[1 * ny * nz + iy * nz + iz] *= 2;
        ids->Jy_flat[(nx - 2) * ny * nz + iy * nz + iz] *= 2;
        ids->Jz_flat[1 * ny * nz + iy * nz + iz] *= 2;
        ids->Jz_flat[(nx - 2) * ny * nz + iy * nz + iz] *= 2;
    }
}


__global__ void applyBCids_nonPeriodicY(struct GPUInterpDensSpecies* ids, struct GPUGrid* gpu_grd, __grid_constant__ const struct parameters param) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int iy = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int iz = blockIdx.z * blockDim.z + threadIdx.z + 1;

    int ny = gpu_grd->nyn;
    int nz = gpu_grd->nzn;
    int nx = gpu_grd->nxn;

    if (!param.PERIODICY && iy == 1 && ix < nx - 1 && iz < nz - 1) {
        ids->rhon_flat[ix * ny * nz + 1 * nz + iz] *= 2;
        ids->rhon_flat[ix * ny * nz + (ny - 2) * nz + iz] *= 2;
        ids->Jx_flat[ix * ny * nz + 1 * nz + iz] *= 2;
        ids->Jx_flat[ix * ny * nz + (ny - 2) * nz + iz] *= 2;
        ids->Jy_flat[ix * ny * nz + 1 * nz + iz] *= 2;
        ids->Jy_flat[ix * ny * nz + (ny - 2) * nz + iz] *= 2;
        ids->Jz_flat[ix * ny * nz + 1 * nz + iz] *= 2;
        ids->Jz_flat[ix * ny * nz + (ny - 2) * nz + iz] *= 2;
    }
}

__global__ void applyBCids_nonPeriodicZ(struct GPUInterpDensSpecies* ids, struct GPUGrid* gpu_grd, __grid_constant__ const struct parameters param) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int iy = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int iz = blockIdx.z * blockDim.z + threadIdx.z + 1;

    int ny = gpu_grd->nyn;
    int nz = gpu_grd->nzn;
    int nx = gpu_grd->nxn;

    if (!param.PERIODICZ && iz == 1 && ix < nx - 1 && iy < ny - 1) {
        ids->rhon_flat[ix * ny * nz + iy * nz + 1] *= 2;
        ids->rhon_flat[ix * ny * nz + iy * nz + (nz - 2)] *= 2;
        ids->Jx_flat[ix * ny * nz + iy * nz + 1] *= 2;
        ids->Jx_flat[ix * ny * nz + iy * nz + (nz - 2)] *= 2;
        ids->Jy_flat[ix * ny * nz + iy * nz + 1] *= 2;
        ids->Jy_flat[ix * ny * nz + iy * nz + (nz - 2)] *= 2;
        ids->Jz_flat[ix * ny * nz + iy * nz + 1] *= 2;
        ids->Jz_flat[ix * ny * nz + iy * nz + (nz - 2)] *= 2;
    }
}

__global__ void sumOverSpeciesNode_kernel(struct GPUInterpDensNet* idn, struct GPUInterpDensSpecies* ids, struct GPUGrid* gpu_grd) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix < gpu_grd->nxn && iy < gpu_grd->nyn && iz < gpu_grd->nzn) {
        int ind = ix * gpu_grd->nyn * gpu_grd->nzn + iy * gpu_grd->nzn + iz;
        idn->rhon_flat[ind] += ids->rhon_flat[ind];
        idn->Jx_flat[ind] += ids->Jx_flat[ind];
        idn->Jy_flat[ind] += ids->Jy_flat[ind];
        idn->Jz_flat[ind] += ids->Jz_flat[ind];
    }
}
