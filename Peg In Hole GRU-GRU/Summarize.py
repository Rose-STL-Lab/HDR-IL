from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

data1 = pd.read_csv("projection a success all.csv")
data2 = pd.read_csv("lift_primitive_data 625 Map large.csv")


data1 = data1[['right_gripper_pole_x_1', 'right_gripper_pole_y_1', 'right_gripper_pole_z_1', 'left_gripper_pole_x_1', 'left_gripper_pole_y_1', 'left_gripper_pole_z_1']].to_numpy()
data2 = data2[['right_gripper_pole_x_1', 'right_gripper_pole_y_1', 'right_gripper_pole_z_1', 'left_gripper_pole_x_1', 'left_gripper_pole_y_1', 'left_gripper_pole_z_1']].to_numpy()





data = np.sqrt(np.sum(np.power((data1 - data2), 2), axis=1))



print(data1[1], data2[1], data[1])

euc = pd.DataFrame(data)

alloutput = []
graspoutput = []
grasp = list(range(0, 10))
sidewaysoutput = []
sideways = list(range(10, 22))
liftoutput = []
lift = list(range(22, 34))
extendoutput = []
extend = list(range(34, 46))
placeoutput = []
place = list(range(46, 58))
retractoutput = []
retract = list(range(58, 70))
allprimitiveoutput = []

for index, row in euc.iterrows():
    res = index % 70

    allprimitiveoutput.append(row)

    #print(row)
    if res in grasp:

        graspoutput.append(row)

    #print(graspoutput)

    if res in sideways:
        sidewaysoutput.append(row)

    if res in lift:
        liftoutput.append(row)

    if res in extend:
        extendoutput.append(row)

    if res in place:
        placeoutput.append(row)

    if res in retract:
        retractoutput.append(row)



alloutput.append(np.mean(graspoutput) )
alloutput.append(np.mean(sidewaysoutput))
alloutput.append(np.mean(liftoutput) )
alloutput.append(np.mean(extendoutput) )
alloutput.append(np.mean(placeoutput) )
alloutput.append(np.mean(retractoutput))

alloutput.append(np.var(graspoutput))
alloutput.append(np.var(sidewaysoutput))
alloutput.append(np.var(liftoutput))
alloutput.append(np.var(extendoutput))
alloutput.append(np.var(placeoutput))
alloutput.append(np.var(retractoutput))

alloutput.append(np.mean(allprimitiveoutput))
alloutput.append(np.var(allprimitiveoutput))



df = pd.DataFrame(alloutput)
df.to_csv("gripper_data.csv")
#print(alloutput)


