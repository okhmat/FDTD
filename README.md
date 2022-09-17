# FDTD

Implementation of Finite Difference Time Domain Method for Maxwell's equations. 
Below is the animation of a wave propagating in 1D along z-axis in free space. Different boundary conditions are illustrated at work. Details of their implementation can be found in the notes. 


Perfect Boundary Condition | Perfect Electric Conductor BC | Perfect Magnetic Conductor BC
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/okhmat/FDTD/blob/main/1D_EyHx_mode_Perfect_BC.gif)  |  ![](https://github.com/okhmat/FDTD/blob/main/1D_EyHx_mode_PEC_BC.gif) | ![](https://github.com/okhmat/FDTD/blob/main/1D_EyHx_mode_PMC_BC.gif)
 PEC BC with hard (voltage) source  |  Periodic BC | Total/Scattered field (one-way) source
![](https://github.com/okhmat/FDTD/blob/main/1D_EyHx_mode_PEC_BC_hard_src.gif)  |  ![](https://github.com/okhmat/FDTD/blob/main/1D_EyHx_mode_Period_BC_1.gif) | ![](https://github.com/okhmat/FDTD/blob/main/1D_EyHx_mode_TF_SF_src.gif)
