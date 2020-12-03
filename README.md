# Deep Imitation Learning for Bimanual Robotic Manipulation

Code for NeurIPS 2020 paper "Deep Imitation Learning for Bimanual Robotic Manipulation"

Code is implemented with PyTorch. Data can be generated using the "Simulation Code"


The prediction models are titled Table_Lift_HDR-IL for the table lifting task, and Peg In Hole HDR-IL for the peg-in-hole task referenced in the paper. Details about training the model can be found in "Model_Readme.txt"

# Generating Data

All code for generating training data is found in the "simulations" folder. Simulations are built up using low-level primitives such as lift or push. All simulations are written using the PyBullet physics engine. All components in the simulations are built-up using URDF files. These files can be easily extended to create other objects/tasks to be used in simulations. In order to save time, simulations are best run on a workstation with a GPU.


# Ground Truth Simulations 
## Table Lifting Simulation                                                 Table Lifting and Connecting Simulation   
![](https://github.com/Rose-STL-Lab/HDR-IL/blob/master/tablelift.gif)       ![](https://github.com/Rose-STL-Lab/HDR-IL/blob/master/tableliftconnect.gif)      

# Citation
If any code from this paper is used in future research, please be sure to include the following citation:
(Add official citation data once available)

# Authors

Fan Xie
Alex Chowdhury








