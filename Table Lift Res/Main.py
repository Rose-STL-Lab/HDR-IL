import math
import statistics

import inline as inline
import matplotlib
import numpy as np
from IPython.display import clear_output
#from tqdm import tqdm_notebook as tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt

import time

import seaborn as sns

import B2ODEModel
import B3TrainODE
import B1DataProcessing

sns.color_palette("bright")
import matplotlib as mpl
import matplotlib.cm as cm
import pandas as pd

import torch
from torch import Tensor, optim
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import B1DataProcessing as dp

import csv
import os
import torch
import torch.nn as nn


import torchvision
import torchvision.transforms as transforms
from torch.utils import data

use_cuda = torch.cuda.is_available()

"""Load all models"""


graspmodel = B3TrainODE.graspmodel


#seqmodel = A3TrainSoftmax.seqmodel


path1 = B3TrainODE.path1





#graspmodel.load_state_dict(torch.load(path1))
#liftmodel.load_state_dict(torch.load(path2))
#extendmodel.load_state_dict(torch.load(path3))
#placemodel.load_state_dict(torch.load(path4))
#retractmodel.load_state_dict(torch.load(path5))
#sidemodel.load_state_dict(torch.load(path6))




"""get data"""

trainsize = 70*1
start = 70*0
datasize = 70
features = 21
train_set = B1DataProcessing.BaxterDataset()

with torch.no_grad():


    outputsize = torch.zeros([70, 1, features])
    outputsize2 = torch.zeros([12, 1, features])

    timetable = []
    for i in range(1, outputsize.size()[0] + 1):
        for j in range(1, outputsize.size()[1] + 1):
            timetable.append(i)

    time = torch.tensor(timetable).view(i, j).unsqueeze(-1).float()

    timetable = []
    for i in range(1, outputsize2.size()[0] + 1):
        for j in range(1, outputsize2.size()[1] + 1):
            timetable.append(i)
    time2 = torch.tensor(timetable).view(i, j).unsqueeze(-1).float()


    visited = torch.zeros([1, 1, features]).cuda()
    truth = torch.zeros([1, 1, features]).cuda()
    varianceslist = torch.zeros([1, 1, features]).cuda()

    #enter as a range
    s= start

    while s < trainsize:
        print(datasize, "datasize", s, trainsize)

        a,b,c,d= train_set[s:s + 1]
        a = a.unsqueeze(1).cuda()
        b = b.unsqueeze(1).cuda()
        c = c.unsqueeze(1).cuda()
        d = d.unsqueeze(1).cuda()



        print(a)
        print(trainsize)

        a1,b1,c1,d1= train_set[s:s + datasize]
        a1 = a1.unsqueeze(1).cuda()
        b1 = b1.unsqueeze(1).cuda()
        c1 = c1.unsqueeze(1).cuda()
        d1 = d1.unsqueeze(1).cuda()

        s = s + datasize
        print(s, "s")

        variances = torch.zeros([1, 1, features]).cuda()





        #while a.size()[0] < 59:

        #print(a.transpose(0, 1).size())
        #primitive = seqmodel.getPrimitive(a.transpose(0, 1).float(), c)
        #while a.size()[0] < 59:
        #    print(a.size())


        #mean, variance = graspmodel.generate_mean_variance(a[-1].unsqueeze(1).float(), time, outputsize)
        #print("mean variance", mean.size(), variance.size())
        #print(mean)
        #print(variance)








        print("size", a.size())
        mean, var = graspmodel.generate_mean_variance(a[-1].unsqueeze(1).float(), outputsize, outputsize)
        print(var)
        print(a.size(), mean.size())
        a = torch.cat((a.float(), mean.float()), 0)
        variances = torch.cat((variances.float(), var.float()), 0)
        print("grasp")


        visited = torch.cat((visited.float(), a[0:datasize, :, :].float()), 0)
        truth = torch.cat((truth.float(), a1[0:datasize, :, :].float()), 0)
        varianceslist = torch.cat((varianceslist.float(), variances[0:datasize, :, :].float()), 0)



        #print(a)


        headers = [

                'right_gripper_pole_x_1',
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

            ]



        #print(a.size())
        #print(a.mean(dim = 2).size())

        #tens = torch.tensor([[[1.0,2.0,3.0]],[[4.0,5.0,6.0]]])
        #print(tens.size())
        #print(tens.mean(dim = 0))


a = a[0:trainsize - start, :, :]
a1 = a1[0:trainsize - start, :, :]
variances = variances[0:trainsize - start, :, :]


with open('projection a.csv', 'w', newline='') as f:
    thewriter = csv.writer(f)
    thewriter.writerow(headers)

    for i in range(1, visited.size()[0]):
        #print(a[i][0].tolist())
        thewriter.writerow(visited[i][0].tolist() + varianceslist[i][0].tolist() +  truth[i][0].tolist())

from matplotlib import pyplot as plt
import numpy as np



plotresults = True

if plotresults == True:

    from matplotlib import pyplot as plt
    import numpy as np

    time = np.arange(trainsize)[0:trainsize - start]


    variances = np.array(variances.tolist())
    variances = np.sqrt(variances)*2


    rx = np.array(a[:, 0, 0].tolist())
    x1 = np.array(a1[:, 0, 0].tolist())
    rxvar = np.array(variances[:, 0, 0])
    rxupper =np.add(rx, rxvar)
    rxlower = np.subtract(rx, rxvar)

    ry = np.array(a[:, 0, 1].tolist())
    y1 = np.array(a1[:, 0, 1].tolist())
    ryvar = np.array(variances[:, 0, 1].tolist())
    ryupper =np.add(ry, ryvar)
    rylower = np.subtract(ry, ryvar)

    rz = np.array(a[:, 0, 2].tolist())
    z1 = np.array(a1[:, 0, 2].tolist())
    rzvar = np.array(variances[:, 0, 2].tolist())
    rzupper =np.add(rz, rzvar)
    rzlower = np.subtract(rz, rzvar)

    lx = np.array(a[:, 0, 7].tolist())
    x2 = np.array(a1[:, 0, 7].tolist())
    lxvar = np.array(variances[:, 0, 7].tolist())
    lxupper =np.add(lx, lxvar)
    lxlower = np.subtract(lx, lxvar)

    ly = np.array(a[:, 0, 8].tolist())
    y2 = np.array(a1[:, 0, 8].tolist())
    lyvar = np.array(variances[:, 0, 8].tolist())
    lyupper =np.add(ly, lyvar)
    lylower = np.subtract(ly, lyvar)

    lz = np.array(a[:, 0, 9].tolist())
    z2 = np.array(a1[:, 0, 9].tolist())
    lzvar = np.array(variances[:, 0, 9].tolist())
    lzupper =np.add(lz, lzvar)
    lzlower = np.subtract(lz, lzvar)





    #plt.plot(time, rx)
    f1 = plt.figure(figsize=(6, 4))
    plt.fill_between(time, rxupper, rxlower, alpha=0.8, label = "Right Predictions")
    plt.fill_between(time, lxupper, lxlower, alpha=0.8, label = "Left Predictions")


    plt.plot(time, x1, 'k--', alpha = .6, label = "Right Demo")
    plt.plot(time, x2, 'r--', alpha = .6, label = "Left Demo")



    plt.title("Single Model X Coordinates", fontsize=18)
    plt.xlabel("Time Steps", fontsize=14)
    plt.ylabel("Meters", fontsize=14)

    #axs[0].plot(time, rx)
    plt.legend(loc = 'upper left')
    plt.tight_layout()
    plt.savefig('x single gnnrnn.png')





    f2 = plt.figure(figsize=(6, 4))
    plt.fill_between(time, ryupper, rylower, alpha=0.8)
    plt.fill_between(time, lyupper, lylower, alpha=0.8)

    plt.plot(time, y1, 'k--', alpha = .6)
    plt.plot(time, y2, 'r--', alpha = .6)



    plt.title("Single Model Y Coordinates", fontsize=18)
    plt.xlabel("Time Steps", fontsize=14)
    plt.ylabel("Meters", fontsize=14)

    # axs[0].plot(time, rx)

    plt.tight_layout()
    plt.savefig('y single gnnrnn.png')


    f3 = plt.figure(figsize=(6, 4))
    plt.fill_between(time, rzupper, rzlower, alpha=0.8)
    plt.fill_between(time, lzupper, lzlower, alpha=0.8)

    plt.plot(time, z1, 'k--', alpha = .6)
    plt.plot(time, z2, 'r--', alpha = .6)



    plt.title("Single Z Coordinates", fontsize=18)
    plt.xlabel("Time Steps", fontsize=14)
    plt.ylabel("Meters", fontsize=14)

    # axs[0].plot(time, rx)

    plt.tight_layout()
    plt.savefig('z single gnnrnn.png')



    plt.show()




