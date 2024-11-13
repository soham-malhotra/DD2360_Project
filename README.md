# DD2360
Applied GPU Programming 

# How to run on Colab 

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

# How to visualize the data

```bash
from google.colab import files
files.download("/content/DD2360/DD2360-PIC/data/rho_net_1.vtk")
```
Next open Paraview
- Open > your/path/to/file/rho_net_1.vtk
- Change the colouring from **Colour** to **rhonet**
- Change the representation from **Outline** to **Slice**

If it all worked out it should look like this : 

![image](https://github.com/user-attachments/assets/cc82d62c-fa57-490d-a367-ed5bb752832d)
