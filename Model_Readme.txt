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
Training code and evaluation code are contained in their own Models. Within each model, there are 7 files relevant to the model.

Main - Runs the models to get the projection. The model is run from this file.
A1PrimitiveData - Get data for planning
A2SoftmaxModel - Prediction models for planning
A3TrainSoftmax - Data processing and training/evaluation for planning model
B1DataProcessing - Get data for dynamic model
B2ODEModel - Prediction models for dynamic model
B3TrainODE - Data processing and training/evaluation for dynamics model

#Run Instructions:
To run the models, run the "Main.py" in each model. All of the other project files are accessed here. There are a few modifications that can be made within the code to select different options.

1. In the A1 and B1 files, make sure the directory in the BaxterDataset constructor is pointing to the correct CSV file.

2. A3 - The Options can be modified for the run. Description of the options are in the comments. 
The options here decide whether to load the previous model or train a new model. 

3. B3 - The Options can be modified for the run. Description of the options are in the comments.
The options here decide whether to load the previous model or train a new model.  

4. Main - Options can be modified for the run. Run this to generate the projections.






