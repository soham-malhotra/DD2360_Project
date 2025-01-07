/** A mixed-precision implicit Particle-in-Cell simulator for heterogeneous systems **/

// Allocator for 2D, 3D and 4D array: chain of pointers
#include "Alloc.h"

// Precision: fix precision for different quantities
#include "PrecisionTypes.h"
// Simulation Parameter - structure
#include "Parameters.h"
#include "GPU_Parameters.h"
// Grid structure
#include "Grid.h"
#include "GPU_Grid.h"

// Interpolated Quantities Structures
#include "InterpDensSpecies.h"
#include "InterpDensNet.h"
#include "GPU_InterpDensSpecies.h"
#include "GPU_InterpDensNet.h"

// Field structure
#include "EMfield.h" // Just E and Bn
#include "EMfield_aux.h" // Bc, Phi, Eth, D
#include "GPU_EMfield.h"

// Particles structure
#include "Particles.h"
#include "Particles_aux.h" // Needed only if dointerpolation on GPU - avoid reduction on GPU
#include "GPU_Particles.h"

// Initial Condition
#include "IC.h"
// Boundary Conditions
#include "BC.h"
// timing
#include "Timing.h"
// Read and output operations
#include "RW_IO.h"

// GPU functions
#include "GPU_Mover.h"
#include "GPU_Interpolator.h"
#include "GPU_DensUtil.h"
#include "GPU_Sort.h"


int main(int argc, char **argv){
    
    // Read the inputfile and fill the param structure
    parameters param;
    // Read the input file name from command line
    readInputFile(&param,argc,argv);
    printParameters(&param);
    saveParameters(&param);
    
    // Timing variables
    double iStart = cpuSecond();
    double iMover, iInterp, eMover = 0.0, eInterp= 0.0;
    
    // Set-up the grid information
    grid grd;
    setGrid(&param, &grd);
    
    // Allocate Fields
    EMfield field;
    field_allocate(&grd,&field);
    EMfield_aux field_aux;
    field_aux_allocate(&grd,&field_aux);
    
    
    // Allocate Interpolated Quantities
    // per species
    interpDensSpecies *ids = new interpDensSpecies[param.ns];
    for (int is=0; is < param.ns; is++)
        interp_dens_species_allocate(&grd,&ids[is],is);
    // Net densities
    interpDensNet idn;
    interp_dens_net_allocate(&grd,&idn);
    
    // Allocate Particles
    particles *part = new particles[param.ns];
    // allocation
    for (int is=0; is < param.ns; is++){
        particle_allocate(&param,&part[is],is);
    }
    
    // Initialization
    initGEM(&param,&grd,&field,&field_aux,part,ids);

    // allocate and copy to GPU
    //TODO make sure copying actually works
    
    GPUGrid* gpu_grd = gpuGridAllocateAndCpy(grd);

    GPUEMfield* gpu_field = gpuFieldAllocate(grd);
    GPUInterpDensNet* gpu_idn = gpuInterpDensNetAllocate(grd);
    GPUInterpDensSpecies** gpu_ids = new GPUInterpDensSpecies*[param.ns];
    GPUParticles** gpu_part = new GPUParticles*[param.ns];  // all pointers to gpu quantities, including per species, live on host
    for (int is=0; is < param.ns; is++){
        gpu_ids[is] = gpuInterpDensSpeciesAllocateAndCpyStatic(grd, ids[is]);
        gpu_part[is] = gpuParticleAllocateAndCpyStatic(grd, part[is]);
    }

    // copy values to gpu
    gpuFieldCpyTo(grd, field, gpu_field);
    gpuInterpDensNetCpyTo(grd, idn, gpu_idn);
    for (int is=0; is < param.ns; is++) {
        gpuInterpDensSpeciesCpyTo(grd, ids[is], gpu_ids[is]);
        gpuParticleCpyTo(part[is], gpu_part[is]);
    }

    // print grid nodes
    printf("Grid cell dimensions are %d x %d x %d\n", grd.nxc, grd.nyc, grd.nzc);
    printf("Grid node dimensions are %d x %d x %d\n", grd.nxn, grd.nyn, grd.nzn);
    
    // **********************************************************//
    // **** Start the Simulation!  Cycle index start from 1  *** //
    // **********************************************************//
    for (int cycle = param.first_cycle_n; cycle < (param.first_cycle_n + param.ncycles); cycle++) {  // TODO remember to also validate with 3d case
        
        std::cout << std::endl;
        std::cout << "***********************" << std::endl;
        std::cout << "   cycle = " << cycle << std::endl;
        std::cout << "***********************" << std::endl;
    
        // set to zero the densities - needed for interpolation
        // setZeroDensities(&idn,ids,&grd,param.ns);
        gpu_setZeroDensities(gpu_ids, gpu_idn, gpu_grd, &param, &grd);

        // for (int is=0; is < param.ns; is++)
        //     mover_PC(&part[is],&field,&grd,&param);
        // iMover = cpuSecond(); // start timer for mover
        gpu_mover_PC(gpu_part, gpu_field, gpu_grd, &part, &param);
        // eMover += (cpuSecond() - iMover); // stop timer for mover
        // std::cout << "   Mover Time (s) = " << eMover << std::endl;

        gpu_sort_particles(gpu_part, part, &param, &grd);

        // interpolation particle to grid
        // iInterp = cpuSecond(); // start timer for the interpolation step
        // for (int is=0; is < param.ns; is++)
        //     interpP2G(&part[is],&ids[is],&grd);
        gpu_interpP2G(gpu_part, gpu_ids, gpu_grd, &part, &grd, &param);
        
        // apply BC to interpolated densities
        // for (int is=0; is < param.ns; is++)
        //     applyBCids(&ids[is],&grd,&param);
        gpu_applyBCids(gpu_ids, gpu_grd, &param, &grd);

        // sum over species
        // sumOverSpecies(&idn,ids,&grd,param.ns);
        gpu_sumOverSpecies(gpu_idn, gpu_ids, gpu_grd, &param, &grd);

        // interpolate charge density from center to node
        // applyBCscalarDensN(idn.rhon,&grd,&param);
        gpu_applyBCscalarDensN(gpu_idn, gpu_grd, &grd, &param);
    }  // end of one PIC cycle

    // copy values to host
    gpuFieldCpyBack(grd, field, gpu_field);
    gpuInterpDensNetCpyBack(grd, idn, gpu_idn);
    for (int is=0; is < param.ns; is++){
        gpuInterpDensSpeciesCpyBack(grd, ids[is], gpu_ids[is]);
        gpuParticleCpyBack(part[is], gpu_part[is]);
    }

    // write E, B, rho to disk
    VTK_Write_Vectors(param.ncycles, &grd,&field);
    VTK_Write_Scalars(param.ncycles, &grd,ids,&idn);

    // clean up on GPU side
    //TODO make sure deallocation actually works. There must be some tool to check for memory leaks?
    gpuGridDeallocate(gpu_grd);
    gpuFieldDeallocate(gpu_field);
    gpuInterpDensNetDeallocate(gpu_idn);
    for (int is=0; is < param.ns; is++){
        gpuInterpDensSpeciesDeallocate(gpu_ids[is]);
        gpuParticleDeallocate(gpu_part[is]);
    }
    // dellocate dynamically stored pointers to GPU
    delete[] gpu_ids;
    delete[] gpu_part;
    
    /// Release the resources
    // deallocate field
    grid_deallocate(&grd);
    field_deallocate(&grd,&field);
    // interp
    interp_dens_net_deallocate(&grd,&idn);
    
    // Deallocate interpolated densities and particles
    for (int is=0; is < param.ns; is++){
        interp_dens_species_deallocate(&grd,&ids[is]);
        particle_deallocate(&part[is]);
    }
    
    
    // stop timer
    double iElaps = cpuSecond() - iStart;
    
    // Print timing of simulation
    std::cout << std::endl;
    std::cout << "**************************************" << std::endl;
    std::cout << "   Tot. Simulation Time (s) = " << iElaps << std::endl;
    std::cout << "   Mover Time / Cycle   (s) = " << eMover/param.ncycles << std::endl;
    std::cout << "   Interp. Time / Cycle (s) = " << eInterp/param.ncycles  << std::endl;
    std::cout << "**************************************" << std::endl;
    
    // exit
    return 0;
}


