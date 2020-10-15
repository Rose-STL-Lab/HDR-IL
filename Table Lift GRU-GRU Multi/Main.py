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

import A1PrimitiveData
import A3TrainSoftmax
import B2Model
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
import A2SoftmaxModel

import torchvision
import torchvision.transforms as transforms
from torch.utils import data

use_cuda = torch.cuda.is_available()

"""Load all models"""


graspmodel = B3TrainODE.graspmodel
liftmodel = B3TrainODE.liftmodel
extendmodel = B3TrainODE.extendmodel
placemodel = B3TrainODE.placemodel
retractmodel = B3TrainODE.retractmodel
sidemodel = B3TrainODE.sidemodel

seqmodel = A3TrainSoftmax.seqmodel


path1 = B3TrainODE.path1
path2 =  B3TrainODE.path2
path3 =  B3TrainODE.path3
path4 =  B3TrainODE.path4
path5 =  B3TrainODE.path5
path6 =  B3TrainODE.path6




#graspmodel.load_state_dict(torch.load(path1))
#liftmodel.load_state_dict(torch.load(path2))
#extendmodel.load_state_dict(torch.load(path3))
#placemodel.load_state_dict(torch.load(path4))
#retractmodel.load_state_dict(torch.load(path5))
#sidemodel.load_state_dict(torch.load(path6))




"""get data"""

trainsize = 70*127
start = 70*0
datasize = 70
features = 21
train_set = B1DataProcessing.BaxterDataset()

with torch.no_grad():


    outputsize = torch.zeros([10, 1, features])
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





        while a.size()[0] < 71:
            c = torch.zeros((a.size()[0], 1, 6))
            primitive = seqmodel.getCurrentPrimitive(a.float(), c)

        #print(a.transpose(0, 1).size())
        #primitive = seqmodel.getPrimitive(a.transpose(0, 1).float(), c)
        #while a.size()[0] < 59:
        #    print(a.size())


        #mean, variance = graspmodel.generate_mean_variance(a[-1].unsqueeze(1).float(), time, outputsize)
        #print("mean variance", mean.size(), variance.size())
        #print(mean)
        #print(variance)

        #test = [0,1,2,3,4, 5]




        #for primitive in test:

            if primitive == 0:
                print("size", a.size())
                mean, var = graspmodel.generate_mean_variance(a[-1].unsqueeze(1).float(), outputsize, outputsize)
                print(var)
                print(a.size(), mean.size())
                a = torch.cat((a.float(), mean.float()), 0)
                variances = torch.cat((variances.float(), var.float()), 0)
                print("grasp")

            #c = torch.zeros((a.size()[0], 1, 6))
            #print(seqmodel.getCurrentPrimitive(a.float(), c))
            if primitive == 1:
                mean, var = sidemodel.generate_mean_variance(a[-1].unsqueeze(1).float(), outputsize2, outputsize2)
                a = torch.cat((a.float(), mean.float()), 0)
                variances = torch.cat((variances.float(), var.float()), 0)
                print("side")

            #c = torch.zeros((a.size()[0], 1, 6))
            #print(seqmodel.getCurrentPrimitive(a.float(), c))
            if primitive == 2:
                mean, var = liftmodel.generate_mean_variance(a[-1].unsqueeze(1).float(), outputsize2, outputsize2)
                a = torch.cat((a.float(), mean.float()), 0)
                variances = torch.cat((variances.float(), var.float()), 0)
                print("lift")
            #c = torch.zeros((a.size()[0], 1, 6))
            #print(seqmodel.getCurrentPrimitive(a.float(), c))
            if primitive == 3:
                mean, var = extendmodel.generate_mean_variance(a[-1].unsqueeze(1).float(), outputsize2, outputsize2)
                a = torch.cat((a.float(), mean.float()), 0)
                variances = torch.cat((variances.float(), var.float()), 0)
                print("extend")
            #c = torch.zeros((a.size()[0], 1, 6))
            #print(seqmodel.getCurrentPrimitive(a.float(), c))
            if primitive == 4:
                mean, var = placemodel.generate_mean_variance(a[-1].unsqueeze(1).float(), outputsize2, outputsize2)
                a = torch.cat((a.float(), mean.float()), 0)
                variances = torch.cat((variances.float(), var.float()), 0)
                print("place")
            #c = torch.zeros((a.size()[0], 1, 6))
            #print(seqmodel.getCurrentPrimitive(a.float(), c))
            if primitive == 5:
                mean, var = retractmodel.generate_mean_variance(a[-1].unsqueeze(1).float(), outputsize2, outputsize2)
                a = torch.cat((a.float(), mean.float()), 0)
                variances = torch.cat((variances.float(), var.float()), 0)
                print( "retract")
            #c = torch.zeros((a.size()[0], 1, 6))
            #print(seqmodel.getCurrentPrimitive(a.float(), c))

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

    plt.axvline(x=10, color='k')
    plt.axvline(x=22, color='k')
    plt.axvline(x=34, color='k')
    plt.axvline(x=46, color='k')
    plt.axvline(x=58, color='k')

    plt.title("Multi-Model X Coordinates", fontsize=18)
    plt.xlabel("Time Steps", fontsize=14)
    plt.ylabel("Meters", fontsize=14)

    #axs[0].plot(time, rx)
    ac = plt.gca()
    ac.set_xticks([5, 17, 29, 41, 53, 65])
    ac.set_xticklabels(["grasp", "move", "lift", "extend", "place", "retract"])

    plt.legend(loc = 'upper left')
    plt.tight_layout()
    plt.savefig('x gnnrnn.png')









    f2 = plt.figure(figsize=(6, 4))
    plt.fill_between(time, ryupper, rylower, alpha=0.8)
    plt.fill_between(time, lyupper, lylower, alpha=0.8)

    plt.plot(time, y1, 'k--', alpha = .6)
    plt.plot(time, y2, 'r--', alpha = .6)

    plt.axvline(x=10, color='k')
    plt.axvline(x=22, color='k')
    plt.axvline(x=34, color='k')
    plt.axvline(x=46, color='k')
    plt.axvline(x=58, color='k')

    plt.title("Multi-Model Y Coordinates", fontsize=18)
    plt.xlabel("Time Steps", fontsize=14)
    plt.ylabel("Meters", fontsize=14)

    # axs[0].plot(time, rx)

    ac = plt.gca()
    ac.set_xticks([5, 17, 29, 41, 53, 65])
    ac.set_xticklabels(["grasp", "move", "lift", "extend", "place", "retract"])

    plt.tight_layout()
    plt.savefig('y gnnrnn.png')


    f3 = plt.figure(figsize=(6, 4))
    plt.fill_between(time, rzupper, rzlower, alpha=0.8)
    plt.fill_between(time, lzupper, lzlower, alpha=0.8)

    plt.plot(time, z1, 'k--', alpha = .6)
    plt.plot(time, z2, 'r--', alpha = .6)

    plt.axvline(x=10, color='k')
    plt.axvline(x=22, color='k')
    plt.axvline(x=34, color='k')
    plt.axvline(x=46, color='k')
    plt.axvline(x=58, color='k')

    plt.title("Multi-Model Z Coordinates", fontsize=18)
    plt.xlabel("Time Steps", fontsize=14)
    plt.ylabel("Meters", fontsize=14)

    # axs[0].plot(time, rx)

    ac = plt.gca()
    ac.set_xticks([5, 17, 29, 41, 53, 65])
    ac.set_xticklabels(["grasp", "move", "lift", "extend", "place", "retract"])
    plt.tight_layout()
    plt.savefig('z gnnrnn.png')



    plt.show()




outputaverage = False

if outputaverage == True:
    data1 = pd.read_csv("projection a.csv")
    data2 = pd.read_csv("lift_primitive_data 625 Map large.csv")

    data1 = data1[['right_gripper_pole_x_1', 'right_gripper_pole_y_1', 'right_gripper_pole_z_1', 'left_gripper_pole_x_1', 'left_gripper_pole_y_1', 'left_gripper_pole_z_1']].to_numpy()
    data2 = data2[['right_gripper_pole_x_1', 'right_gripper_pole_y_1', 'right_gripper_pole_z_1', 'left_gripper_pole_x_1', 'left_gripper_pole_y_1', 'left_gripper_pole_z_1']].to_numpy()

    data = np.sqrt(np.sum(np.power((data1 - data2), 2), axis=1))
    data = pd.DataFrame(data)

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

    for index, row in data.iterrows():
        res = index % 70
        if res in grasp:
            graspoutput.append(row)

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

    alloutput.append(sum(graspoutput) / len(graspoutput))
    alloutput.append(sum(sidewaysoutput) / len(sidewaysoutput))
    alloutput.append(sum(liftoutput) / len(liftoutput))
    alloutput.append(sum(extendoutput) / len(extendoutput))
    alloutput.append(sum(placeoutput) / len(placeoutput))
    alloutput.append(sum(retractoutput) / len(retractoutput))

    alloutput.append(statistics.variance(graspoutput))
    alloutput.append(statistics.variance(sidewaysoutput))
    alloutput.append(statistics.variance(liftoutput))
    alloutput.append(statistics.variance(extendoutput))
    alloutput.append(statistics.variance(placeoutput))
    alloutput.append(statistics.variance(retractoutput))

    df = pd.DataFrame(alloutput)
    df.to_csv("gripper_data.csv")
    print(alloutput)


