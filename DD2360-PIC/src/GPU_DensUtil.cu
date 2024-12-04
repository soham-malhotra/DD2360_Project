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

void gpu_applyBCids(struct GPUInterpDensSpecies** gpu_ids,  struct GPUGrid* gpu_grd, parameters* param, struct grid* grd) {

    dim3 dimBlock(THREAD_NR_PER_X, THREAD_NR_PER_Y, THREAD_NR_PER_Z);
    dim3 dimGrid(ceil(grd->nxn/THREAD_NR_PER_X), ceil(grd->nyn/THREAD_NR_PER_Y), ceil(grd->nzn/THREAD_NR_PER_Z));

    for (int is = 0; is < param->ns; is++) {
        applyBCids_kernel<<<dimGrid, dimBlock>>>(gpu_ids[is], gpu_grd, *param);
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

__global__ void applyBCids_kernel(struct GPUInterpDensSpecies* ids, struct GPUGrid* gpu_grd, __grid_constant__ const struct parameters param) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int iy = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int iz = blockIdx.z * blockDim.z + threadIdx.z + 1;

    int nx = gpu_grd->nxn;
    int ny = gpu_grd->nyn;
    int nz = gpu_grd->nzn;

    if (ix < gpu_grd->nxn - 1 && iy < gpu_grd->nyn - 1 && iz < gpu_grd->nzn - 1) {
        if (param.PERIODICX==true) {
            ids->rhon_flat[1 * ny * nz + iy * nz + iz] += ids->rhon_flat[(nx-2) * ny * nz + iy * nz + iz];  // surely the compiler optimizes these?
            ids->rhon_flat[(nx - 2) * ny * nz + iy * nz + iz] = ids->rhon_flat[1 * ny * nz + iy * nz + iz];
            ids->Jx_flat[1 * ny * nz + iy * nz + iz] += ids->Jx_flat[(nx-2) * ny * nz + iy * nz + iz];
            ids->Jx_flat[(nx - 2) * ny * nz + iy * nz + iz] = ids->Jx_flat[1 * ny * nz + iy * nz + iz];
            ids->Jy_flat[1 * ny * nz + iy * nz + iz] += ids->Jy_flat[(nx-2) * ny * nz + iy * nz + iz];
            ids->Jy_flat[(nx - 2) * ny * nz + iy * nz + iz] = ids->Jy_flat[1 * ny * nz + iy * nz + iz];
            ids->Jz_flat[1 * ny * nz + iy * nz + iz] += ids->Jz_flat[(nx-2) * ny * nz + iy * nz + iz];
            ids->Jz_flat[(nx - 2) * ny * nz + iy * nz + iz] = ids->Jz_flat[1 * ny * nz + iy * nz + iz];
        }

        if (param.PERIODICY==true) {
            ids->rhon_flat[ix * ny * nz + 1 * nz + iz] += ids->rhon_flat[ix * ny * nz + (ny-2) * nz + iz];
            ids->rhon_flat[ix * ny * nz + (ny-2) * nz + iz] = ids->rhon_flat[ix * ny * nz + 1 * nz + iz];
            ids->Jx_flat[ix * ny * nz + 1 * nz + iz] += ids->Jx_flat[ix * ny * nz + (ny-2) * nz + iz];
            ids->Jx_flat[ix * ny * nz + (ny-2) * nz + iz] = ids->Jx_flat[ix * ny * nz + 1 * nz + iz];
            ids->Jy_flat[ix * ny * nz + 1 * nz + iz] += ids->Jy_flat[ix * ny * nz + (ny-2) * nz + iz];
            ids->Jy_flat[ix * ny * nz + (ny-2) * nz + iz] = ids->Jy_flat[ix * ny * nz + 1 * nz + iz];
            ids->Jz_flat[ix * ny * nz + 1 * nz + iz] += ids->Jz_flat[ix * ny * nz + (ny-2) * nz + iz];
            ids->Jz_flat[ix * ny * nz + (ny-2) * nz + iz] = ids->Jz_flat[ix * ny * nz + 1 * nz + iz];
        }

        if (param.PERIODICZ==true) {
            ids->rhon_flat[ix * ny * nz + iy * nz + 1] += ids->rhon_flat[ix * ny * nz + iy * nz + (nz-2)];
            ids->rhon_flat[ix * ny * nz + iy * nz + (nz-2)] = ids->rhon_flat[ix * ny * nz + iy * nz + 1];
            ids->Jx_flat[ix * ny * nz + iy * nz + 1] += ids->Jx_flat[ix * ny * nz + iy * nz + (nz-2)];
            ids->Jx_flat[ix * ny * nz + iy * nz + (nz-2)] = ids->Jx_flat[ix * ny * nz + iy * nz + 1];
            ids->Jy_flat[ix * ny * nz + iy * nz + 1] += ids->Jy_flat[ix * ny * nz + iy * nz + (nz-2)];
            ids->Jy_flat[ix * ny * nz + iy * nz + (nz-2)] = ids->Jy_flat[ix * ny * nz + iy * nz + 1];
            ids->Jz_flat[ix * ny * nz + iy * nz + 1] += ids->Jz_flat[ix * ny * nz + iy * nz + (nz-2)];
            ids->Jz_flat[ix * ny * nz + iy * nz + (nz-2)] = ids->Jz_flat[ix * ny * nz + iy * nz + 1];
        }

        if (param.PERIODICX==false) {
            ids->rhon_flat[1 * ny * nz + iy * nz + iz] *= 2;
            ids->rhon_flat[(nx-2) * ny * nz + iy * nz + iz] *= 2;
            ids->Jx_flat[1 * ny * nz + iy * nz + iz] *= 2;
            ids->Jx_flat[(nx-2) * ny * nz + iy * nz + iz] *= 2;
            ids->Jy_flat[1 * ny * nz + iy * nz + iz] *= 2;
            ids->Jy_flat[(nx-2) * ny * nz + iy * nz + iz] *= 2;
            ids->Jz_flat[1 * ny * nz + iy * nz + iz] *= 2;
            ids->Jz_flat[(nx-2) * ny * nz + iy * nz + iz] *= 2;
        }

        if (param.PERIODICY==false) {
            ids->rhon_flat[ix * ny * nz + 1 * nz + iz] *= 2;
            ids->rhon_flat[ix * ny * nz + (ny-2) * nz + iz] *= 2;
            ids->Jx_flat[ix * ny * nz + 1 * nz + iz] *= 2;
            ids->Jx_flat[ix * ny * nz + (ny-2) * nz + iz] *= 2;
            ids->Jy_flat[ix * ny * nz + 1 * nz + iz] *= 2;
            ids->Jy_flat[ix * ny * nz + (ny-2) * nz + iz] *= 2;
            ids->Jz_flat[ix * ny * nz + 1 * nz + iz] *= 2;
            ids->Jz_flat[ix * ny * nz + (ny-2) * nz + iz] *= 2;
        }

        if (param.PERIODICZ==false) {
            ids->rhon_flat[ix * ny * nz + iy * nz + 1] *= 2;
            ids->rhon_flat[ix * ny * nz + iy * nz + (nz-2)] *= 2;
            ids->Jx_flat[ix * ny * nz + iy * nz + 1] *= 2;
            ids->Jx_flat[ix * ny * nz + iy * nz + (nz-2)] *= 2;
            ids->Jy_flat[ix * ny * nz + iy * nz + 1] *= 2;
            ids->Jy_flat[ix * ny * nz + iy * nz + (nz-2)] *= 2;
            ids->Jz_flat[ix * ny * nz + iy * nz + 1] *= 2;
            ids->Jz_flat[ix * ny * nz + iy * nz + (nz-2)] *= 2;
        }
    }
}
