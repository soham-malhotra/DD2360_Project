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
};

/**
 * @brief: allocates memory on device and copies data from host to device
 * @param: gpu_grid -> pointer to the pointer that points to the grid on device memory
 * @param: grid -> pointer to host memory
 */
struct GPUgrid* gpuGridAllocateAndCpy(const struct grid&);

/**
 * @brief: deallocates memory allocated for GPUGrid
 * @param: gpu_grid -> the grid to deallocate
 */
void gpuGridDeallocate(struct GPUgrid*);

/**
 * @brief: assigns all static members of grid to gpu_grid
 * @param: gpu_grid -> reference to grid object on host
 */
void copyStaticMembers(const grid& grd, GPUgrid& gpu_grd);

#endif
