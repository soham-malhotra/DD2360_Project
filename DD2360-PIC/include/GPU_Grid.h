#ifndef GPU_GRID_H
#define GPU_GRID_H

#include "Grid.h"
#include "cudaAux.h"

/** Grid Data */
struct GPUgrid {
    /** number of cells - X direction, including + 2 (guard cells) */
    int nxc;
    /** number of nodes - X direction, including + 2 extra nodes for guard cells */
    int nxn;
    /** number of cell - Y direction, including + 2 (guard cells) */
    int nyc;
    /** number of nodes - Y direction, including + 2 extra nodes for guard cells */
    int nyn;
    /** number of cell - Z direction, including + 2 (guard cells) */
    int nzc;
    /** number of nodes - Z direction, including + 2 extra nodes for guard cells */
    int nzn;
    /** dx = space step - X direction */
    double dx;
    /** dy = space step - Y direction */
    double dy;
    /** dz = space step - Z direction */
    double dz;
    /** invdx = 1/dx */
    FPfield invdx;
    /** invdy = 1/dy */
    FPfield invdy;
    /** invdz = 1/dz */
    FPfield invdz;
    /** invol = inverse of volume*/
    FPfield invVOL;
    /** local grid boundaries coordinate  */
    double xStart, xEnd, yStart, yEnd, zStart, zEnd;
    /** domain size */
    double Lx, Ly, Lz;
    
    /** Periodicity for fields X **/
    bool PERIODICX;
    /** Periodicity for fields Y **/
    bool PERIODICY;
    /** Periodicity for fields Z **/
    bool PERIODICZ;
    
    // Nodes coordinate
    /** coordinate node X */
    FPfield* XN_GPU_flat;
    /** coordinate node Y */
    FPfield* YN_GPU_flat;
    /** coordinate node Z */
    FPfield* ZN_GPU_flat;

    // Assignment operator to copy from a grid object
    /** equality operator to copy */
    GPUgrid& operator=(const grid& src) {
        nxc = src.nxc;
        nxn = src.nxn;
        nyc = src.nyc;
        nyn = src.nyn;
        nzc = src.nzc;
        nzn = src.nzn;

        dx = src.dx;
        dy = src.dy;
        dz = src.dz;

        invdx = src.invdx;
        invdy = src.invdy;
        invdz = src.invdz;
        invVOL = src.invVOL;

        xStart = src.xStart;
        xEnd = src.xEnd;
        yStart = src.yStart;
        yEnd = src.yEnd;
        zStart = src.zStart;
        zEnd = src.zEnd;

        Lx = src.Lx;
        Ly = src.Ly;
        Lz = src.Lz;

        PERIODICX = src.PERIODICX;
        PERIODICY = src.PERIODICY;
        PERIODICZ = src.PERIODICZ;

        size_t size = nxn * nyn * nzn;

        cudaErrorHandling(cudaMalloc(XN_GPU_flat, size));
        cudaErrorHandling(cudaMalloc(YN_GPU_flat, size));
        cudaErrorHandling(cudaMalloc(ZN_GPU_flat, size));

        cudaErrorHandling(cudaMemCpy(XN_GPU_flat, src.XN_flat, size, MemcpyHostToDevice));
        cudaErrorHandling(cudaMemCpy(YN_GPU_flat, src.YN_flat, size, MemcpyHostToDevice));
        cudaErrorHandling(cudaMemCpy(ZN_GPU_flat, src.ZN_flat, size, MemcpyHostToDevice));

        return *this;
    }


};

/**
 * @brief: deallocates memory allocated for GPUGrid
 * @param: gpu_grid -> the grid to deallocate
 */
void deallocateGPUGrid(GPUgrid* gpu_grid) {
}

#endif
