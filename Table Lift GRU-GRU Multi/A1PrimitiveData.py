from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import pandas as pd



def findFiles(path): return glob.glob(path)


# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [line for line in lines]


def getSampleNN(tensor):
    column = 0
    row = 0

    x_val = tensor[column, row]





def splitTrainingTesting(data, size):
    perm = torch.randperm(data.size(0))
    idx = perm[:size]
    samples = data[idx]

    return samples


"""split data into actions and object coordinates"""


def splitDataCoordinatesActions(data):
    coordinates1 = data[:, 0:21]
    coordinates2 = data[:, 21:42]


    return coordinates1, coordinates2


class BaxterDataset(Dataset):

    def __init__(self):

        #filePathTrain = '../../PrimittiveData4.4.2020/primitive data side movement 12 Step filtering.csv'
        #filePathTrain = '../../PrimittiveData4.10.2020/primitive data filtering.csv'
        filePathTrain = '../../PrimitiveData4.23.20/primitive data filtering 625 L.csv'

        xy = pd.read_csv(filePathTrain)
        self.xraw, self.yraw = selectRowsRNN6Locations(xy)
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


def selectRowsRNN6Forces(data):
    features = data[['force_1',
                     'force_2',
                     'force_3',
                     'force_4',
                     'force_5',
                     'force_6',

                     'left_back_corner_x1',
                     'left_back_corner_y1',
                     'left_back_corner_z1',

                     'left_front_corner_x1',
                     'left_front_corner_y1',
                     'left_front_corner_z1',
                     'right_back_corner_x1',
                     'right_back_corner_y1',
                     'right_back_corner_z1',
                     'right_front_corner_x1',
                     'right_front_corner_y1',
                     'right_front_corner_z1',

                     ]]

    labels = data[[
        'left_back_corner_x2',
        'left_back_corner_y2',
        'left_back_corner_z2',

        'left_front_corner_x2',
        'left_front_corner_y2',
        'left_front_corner_z2',
        'right_back_corner_x2',
        'right_back_corner_y2',
        'right_back_corner_z2',
        'right_front_corner_x2',
        'right_front_corner_y2',
        'right_front_corner_z2'
    ]]
    return features, labels


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

                     # 'left_upper_forearm_quat1_1',
                     # 'left_upper_forearm_quat2_1',
                     # 'left_upper_forearm_quat3_1',
                     # 'left_upper_forearm_quat4_1',
                     # 'left_upper_forearm_x_1',
                     # 'left_upper_forearm_y_1',
                     # 'left_upper_forearm_z_1',
                     # 'right_upper_forearm_quat1_1',
                     # 'right_upper_forearm_quat2_1',
                     # 'right_upper_forearm_quat3_1',
                     # 'right_upper_forearm_quat4_1',
                     # 'right_upper_forearm_x_1',
                     # 'right_upper_forearm_y_1',
                     # 'right_upper_forearm_z_1',

                     # 'left_lower_forearm_quat1_1',
                     # 'left_lower_forearm_quat2_1',
                     # 'left_lower_forearm_quat3_1',
                     # 'left_lower_forearm_quat4_1',
                     # 'left_lower_forearm_x_1',
                     # 'left_lower_forearm_y_1',
                     # 'left_lower_forearm_z_1',
                     # 'right_lower_forearm_quat1_1',
                     # 'right_lower_forearm_quat2_1',
                     # 'right_lower_forearm_quat3_1',
                     # 'right_lower_forearm_quat4_1',
                     # 'right_lower_forearm_x_1',
                     # 'right_lower_forearm_y_1',
                     # 'right_lower_forearm_z_1',
                     # 'left_wrist_quat1_1',
                     # 'left_wrist_quat2_1',
                     # 'left_wrist_quat3_1',
                     # 'left_wrist_quat4_1',

                     'x_1',
                     'y_1',
                     'z_1',

                     'quat1_1',
                     'quat2_1',
                     'quat3_1',
                     'quat4_1',

                     # 'EndRX',
                     # 'EndRY',
                     # 'EndRZ',
                     # 'EndLX',
                     # 'EndLY',
                     # 'EndLZ',

                     # 'left_back_corner_x_1',
                     # 'left_back_corner_y_1',
                     # 'left_back_corner_z_1',

                     # 'left_front_corner_x_1',
                     # 'left_front_corner_y_1',
                     # 'left_front_corner_z_1',
                     # 'right_back_corner_x_1',
                     # 'right_back_corner_y_1',
                     # 'right_back_corner_z_1',
                     # 'right_front_corner_x_1',
                     # 'right_front_corner_y_1',
                     # 'right_front_corner_z_1',

                     # 'left_wrist_quat1_1',
                     # 'left_wrist_quat2_1',
                     # 'left_wrist_quat3_1',
                     # 'left_wrist_quat4_1',
                     # 'right_wrist_quat1_1',
                     # 'right_wrist_quat2_1',
                     # 'right_wrist_quat3_1',
                     # 'right_wrist_quat4_1',
                     # 'gripper_open',

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
                     #'gripper_open',

                     # 'left_wrist_quat1_2',
                     # 'left_wrist_quat2_2',
                     # 'left_wrist_quat3_2',
                     # 'left_wrist_quat4_2',

                     # 'left_upper_forearm_quat1_2',
                     # 'left_upper_forearm_quat2_2',
                     # 'left_upper_forearm_quat3_2',
                     # 'left_upper_forearm_quat4_2',
                     # 'left_upper_forearm_x_2',
                     # 'left_upper_forearm_y_2',
                     # 'left_upper_forearm_z_2',
                     # 'right_upper_forearm_quat1_2',
                     # 'right_upper_forearm_quat2_2',
                     # 'right_upper_forearm_quat3_2',
                     # 'right_upper_forearm_quat4_2',
                     # 'right_upper_forearm_x_2',
                     # 'right_upper_forearm_y_2',
                     # 'right_upper_forearm_z_2',

                     # 'left_lower_forearm_quat1_2',
                     # 'left_lower_forearm_quat2_2',
                     # 'left_lower_forearm_quat3_2',
                     # 'left_lower_forearm_quat4_2',
                     # 'left_lower_forearm_x_2',
                     # 'left_lower_forearm_y_2',
                     # 'left_lower_forearm_z_2',
                     # 'right_lower_forearm_quat1_2',
                     # 'right_lower_forearm_quat2_2',
                     # 'right_lower_forearm_quat3_2',
                     # 'right_lower_forearm_quat4_2',
                     # 'right_lower_forearm_x_2',
                     # 'right_lower_forearm_y_2',
                     # 'right_lower_forearm_z_2',

                     # 'left_back_corner_x_2',
                     # 'left_back_corner_y_2',
                     # 'left_back_corner_z_2',

                     # 'left_front_corner_x_2',
                     # 'left_front_corner_y_2',
                     # 'left_front_corner_z_2',
                     # 'right_back_corner_x_2',
                     # 'right_back_corner_y_2',
                     # 'right_back_corner_z_2',
                     # 'right_front_corner_x_2',
                     # 'right_front_corner_y_2',
                     # 'right_front_corner_z_2',

                     # 'left_wrist_quat1_2',
                     # 'left_wrist_quat2_2',
                     # 'left_wrist_quat3_2',
                     # 'left_wrist_quat4_2',
                     # 'right_wrist_quat1_2',
                     # 'right_wrist_quat2_2',
                     # 'right_wrist_quat3_2',
                     # 'right_wrist_quat4_2',
                     # 'gripper_open',

                     ]]

    labels = data[[

        'primitive1',
        'primitive2',
        'primitive3',
        'primitive4',
        'primitive5',

        'primitive6'
    ]]

    return features, labels
