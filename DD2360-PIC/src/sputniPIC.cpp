/** A mixed-precision implicit Particle-in-Cell simulator for heterogeneous systems **/

// Allocator for 2D, 3D and 4D array: chain of pointers
#include "Alloc.h"

// Precision: fix precision for different quantities
#include "PrecisionTypes.h"
// Simulation Parameter - structure
#include "Parameters.h"
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
#include "GPU_EMfield_aux.h"

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


    GPUgrid* gpu_grid;
    GPUEMfield* gpu_field;
    GPUInterpDensNet* gpu_idn;

    // allocate array of pointers that point to pointers to GPUInterpDensSpecies on device
    GPUInterpDensSpecies** gpu_ids;  
    cudaErrorHandling(cudaMalloc(&gpu_ids, param.ns * sizeof(GPUInterpDensSpecies*)));

    // allocate array of pointers that point to pointers to GPUParticles on device
    GPUParticles** gpu_part; 
    cudaErrorHandling(cudaMalloc(&gpu_part, param.ns * sizeof(GPUParticles*))); 

    // allocate and copy to GPU
    //TODO initialize flat arrays on CPU side by initGEM
    //TODO make sure copying actually works
    gpuGridAllocateAndCpy(grd, gpu_grid);
    gpuFieldAllocateAndCpy(grd, gpu_field, field);
    gpuInterpDensNetAllocateAndCpy(grd, gpu_idn, idn);
    for (int is=0; is < param.ns; is++){
        gpuInterpDensSpeciesAllocateAndCpy(grd, gpu_ids[is], ids[is]);
        gpuParticleAllocateAndCpy(grd, gpu_part[is], part[is]);
    }
    
    
    // **********************************************************//
    // **** Start the Simulation!  Cycle index start from 1  *** //
    // **********************************************************//
    for (int cycle = param.first_cycle_n; cycle < (param.first_cycle_n + param.ncycles); cycle++) {
        
        std::cout << std::endl;
        std::cout << "***********************" << std::endl;
        std::cout << "   cycle = " << cycle << std::endl;
        std::cout << "***********************" << std::endl;
    
        // set to zero the densities - needed for interpolation
        setZeroDensities(&idn,ids,&grd,param.ns);
        

        // implicit mover
        iMover = cpuSecond(); // start timer for mover
        for (int is=0; is < param.ns; is++)
            mover_PC(&part[is],&field,&grd,&param);
        eMover += (cpuSecond() - iMover); // stop timer for mover
        
        
        
        
        // interpolation particle to grid
        iInterp = cpuSecond(); // start timer for the interpolation step
        // interpolate species
        for (int is=0; is < param.ns; is++)
            interpP2G(&part[is],&ids[is],&grd);
        // apply BC to interpolated densities
        for (int is=0; is < param.ns; is++)
            applyBCids(&ids[is],&grd,&param);
        // sum over species
        sumOverSpecies(&idn,ids,&grd,param.ns);
        // interpolate charge density from center to node
        applyBCscalarDensN(idn.rhon,&grd,&param);
        
        
        
        // write E, B, rho to disk
        if (cycle%param.FieldOutputCycle==0){
            VTK_Write_Vectors(cycle, &grd,&field);
            VTK_Write_Scalars(cycle, &grd,ids,&idn);
        }
        
        eInterp += (cpuSecond() - iInterp); // stop timer for interpolation
        
        
    
    }  // end of one PIC cycle

    // clean up on GPU side
    //TODO clean up flat arrays on CPU side
    gpuGridDeallocate(gpu_grid);
    gpuFieldDeallocate(gpu_field);
    gpuInterpDensNetDeallocate(gpu_idn);
    for (int is=0; is < param.ns; is++){
        gpuInterpDensSpeciesDeallocate(gpu_ids[is]);
        gpuParticleDeallocate(gpu_part[is]);
    }
    
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


