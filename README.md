# DD2360 Applied GPU Programming
Implementation of a tiled PIC that uses CUDA API to utilise GPU hardware to accelerate particle interpolation and moving. The moving is done using the Boris-Leapfrog Algorithm, enhanced by having a thread handle these expensive computation per particle. Furthermore, streams are utilised to handle the different species launched. Sorting the particles with respect to the cell in which they reside improves performance as it allows for tiling and utilising shared memory in both the interpolator and the mover. A diagram is given below. NSight and ncu tools were used for profiling. 

![image](https://github.com/user-attachments/assets/5c072f77-879e-4d09-a8ee-787694f56c46)


# How to run the code

```bash

!git clone https://github.com/KTH-ScaLab/DD2360.git

%cd DD2360/DD2360-PIC/

%mkdir data

!ls

!make

!./bin/sputniPIC.out an_input_file

```

The input file is either : 
- *inputfiles/GEM_3D.inp*
- *inputfiles/GEM_2D.inp*

# How to visualize the data (instructions for Google Colab)

```bash
from google.colab import files
files.download("/content/DD2360/DD2360-PIC/data/rho_net_1.vtk")
```
If the code is run on the machine locally, just follow the instructions below 

Next open Paraview
- Open > your/path/to/file/rho_net_1.vtk (note it can be any rho_net_[integer].vtk, as these are just different cycle runs)
- Change the colouring from **Colour** to **rhonet**
- Change the representation from **Outline** to **Slice**

If it all worked out it should look like this : 

![image](https://github.com/user-attachments/assets/cc82d62c-fa57-490d-a367-ed5bb752832d)
