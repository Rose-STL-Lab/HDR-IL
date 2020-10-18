from __future__ import unicode_literals, print_function, division
from io import open

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import pandas as pd






def splitTrainingTesting(data, size):
    perm = torch.randperm(data.size(0))
    idx = perm[:size]
    samples = data[idx]

    return samples



"""split data into actions and object coordinates"""
def splitDataCoordinatesActions(data):

    #coordinates1 = data[:, 0:34]
    #coordinates2 = data[:, 34:68]

    coordinates1 = data[:, 0:21]
    coordinates2 = data[:, 21:42]

    return  coordinates1, coordinates2



class BaxterDataset(Dataset):

    def __init__(self):

        #filePathTrain = '../../PrimitiveData4.23.20/primitive data filtering 625 L.csv'
        filePathTrain = '../../PrimitiveData4.23.20/primitive data random sample.csv'


        xy = pd.read_csv(filePathTrain)
        self.xraw, self.yraw= selectRowsRNN6Locations(xy)
        self.x = torch.tensor(self.xraw.values)


        self.c, self.f = splitDataCoordinatesActions(self.x)


        self.y = torch.tensor(self.yraw.values)

        time = []
        for i in range(0, self.x.size()[0]):
            time.append(i)
        self.t = torch.tensor(time).unsqueeze(1)


    def __getitem__(self, index):
        return self.c[index], self.f[index], self.y[index], self.t[index]

    def __len__(self):
        return self.c.size()[0]




"""Model can predict the mass"""
def selectRowsRNN6Locations(data):
    features = data[['right_gripper_pole_x_1',
                     'right_gripper_pole_y_1',
                     'right_gripper_pole_z_1',

                     'right_gripper_pole_q_11',
                     'right_gripper_pole_q_12',
                     'right_gripper_pole_q_13',
                     'right_gripper_pole_q_14',

                     'left_gripper_pole_x_1',
                     'left_gripper_pole_y_1',
                     'left_gripper_pole_z_1',
                     'left_gripper_pole_q_11',
                     'left_gripper_pole_q_12',
                     'left_gripper_pole_q_13',
                     'left_gripper_pole_q_14',


                     'x_1',
                     'y_1',
                     'z_1',
                     'quat1_1',
                     'quat2_1',
                     'quat3_1',
                     'quat4_1',




                     'right_gripper_pole_x_2',
                     'right_gripper_pole_y_2',
                     'right_gripper_pole_z_2',
                     'right_gripper_pole_q_21',
                     'right_gripper_pole_q_22',
                     'right_gripper_pole_q_23',
                     'right_gripper_pole_q_24',

                     'left_gripper_pole_x_2',
                     'left_gripper_pole_y_2',
                     'left_gripper_pole_z_2',
                     'left_gripper_pole_q_21',
                     'left_gripper_pole_q_22',
                     'left_gripper_pole_q_23',
                     'left_gripper_pole_q_24',

                     'x_2',
                     'y_2',
                     'z_2',
                     'quat1_2',
                     'quat2_2',
                     'quat3_2',
                     'quat4_2',


                     ]]

    labels = data[[

        'right_gripper_pole_x_1',
        'right_gripper_pole_y_1',
        'right_gripper_pole_z_1',
        'left_gripper_pole_x_1',
        'left_gripper_pole_y_1',
        'left_gripper_pole_z_1',

        'x_1',
        'y_1',
        'z_1',

        'quat1_1',
        'quat2_1',
        'quat3_1',
        'quat4_1',
        'Index',
    ]]



    return features, labels
