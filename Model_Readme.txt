#Model Dependencies:

###Simulation Models
- bumpy
- pandas
- pybullet


###Projection Models
- statistics
- torch
- pandas
- numpy
- random
- csv
- dgl

#Runtime:
Models were trained on a laptop with an i7-8750H processor and Nvidia 1050Ti Max-Q GPU.
2500 demonstrations for the table lift task takes about 2 hours to run
4500 demonstrations for the peg-in-hole task takes about 3.5 hours to run


###Relevant Files

TrainDynamicModels - Train all primitive dynamics models.
TrainPlanningModels - Train the planning model.
GenerateProjections - Run this with trained models. The directories of models are loaded from the training files.
Models\HDRIL_Models - Contains the RNN models used for the dynamics and planning models.

#Run Instructions:

1. Update the directory in the BaxterDataset() function in utils to point to the simulation data.
2. In the TrainDynamicsModel, update parameters at top of the file. Run to generate trained dynamics models.
3. In the TrainPlanningModels, update parameters at top of the file. Run to generate trained planning model.
The options here decide whether to load the previous model or train a new model. 
4. Run the GenerateProjections file with the appropriate parameters to generate projections.







