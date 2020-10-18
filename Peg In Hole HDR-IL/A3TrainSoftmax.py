import math

import inline as inline
import matplotlib
import numpy as np
from IPython.display import clear_output

from random import randint
import random


import matplotlib as mpl
import matplotlib.pyplot as plt

import B1DataProcessing
import NNModel
import time
import A2SoftmaxModel

import seaborn as sns

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
import B2Model

import torchvision
import torchvision.transforms as transforms
from torch.utils import data
import A1PrimitiveData

import dgl

use_cuda = torch.cuda.is_available()

"""Read in the files"""

runs = 4700
runsize = 130
load_model = True
train_model = False

trainsize = runs*runsize

train_set = A1PrimitiveData.BaxterDataset()

trainingdata = data.Subset(train_set, indices=list(range(0, trainsize)))

testdata = data.Subset(train_set, indices=list(range(trainsize, 10000)))

sequencelength = 130

train_loader = torch.utils.data.DataLoader(trainingdata, shuffle=False, batch_size=sequencelength)
test_loader = torch.utils.data.DataLoader(testdata, shuffle=False, batch_size=sequencelength)

input_size = train_set.c.size()[1]
input_size2 = train_set.f.size()[1]

output_size = train_set.y.size()[1]

"""The model"""
n_hidden = 28
n_latent = 512
n_iters = 20

current_loss = 0
test_loss_total = 0
all_losses = []
test_losses = []
losses = []

force = []

n_epochs = 2



activation = nn.Sequential()



g2 = dgl.DGLGraph()
nodes = 28
totalnodes2 = 28
g2.add_nodes(nodes)
# A couple edges one-by-one
for i in range(0, nodes):
    for j in range(0, nodes):
        g2.add_edge(i, j)




gcn1 = A2SoftmaxModel.GAT(g2, int(input_size/totalnodes2), n_latent, n_hidden, 1)

encoder = A2SoftmaxModel.RNNEncoder(input_size, output_size, n_latent)
decoder = A2SoftmaxModel.NeuralODEDecoder(input_size, output_size, n_latent)

encoder_optimizer = optim.Adam(gcn1.parameters(), lr=.00005)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=.00005)
# criterion = nn.MSELoss()
# criterion = nn.BCELoss()
criterion = nn.CrossEntropyLoss()
#criterion = nn.NLLLoss()
seqmodel = A2SoftmaxModel.ODEVAE(input_size, output_size, n_latent, gcn1, decoder, encoder_optimizer,
                                 decoder_optimizer,
                                 criterion)

seqmodel = seqmodel.cuda()

"""Model training"""

PATH = 'model2stepnonoise.pt'



if (load_model == True):
    seqmodel.load_state_dict(torch.load(PATH))

if (train_model == True):

    for epoch_idx in range(n_epochs):

        print(epoch_idx)

        i = 0

        while i < trainsize:

            primitive = train_set.f[-1][-1]

            s = random.randint(0, runs - 1) * runsize


            sequencelength = 130


            length = sequencelength
            #length = randint(1, 58)
            c, f, y, t = train_set[s:s + length]
            i += sequencelength




            rows = int(c.size()[0] / length)

            startcord = c.reshape(rows, length, input_size).transpose(0, 1).cuda()
            endcord = f.reshape(rows, length, input_size2).transpose(0, 1).cuda()

            force = y.reshape(rows, length, output_size).transpose(0, 1).cuda()


            target = t.reshape(length, rows, 1).cuda()


            trainloss = 0
            testloss = 0


            avgloss = seqmodel.train(startcord.float(), force.float())

            trainavgloss = avgloss

            #print(force.size())

            print("comparison", seqmodel.getPrimitive(startcord.float(), force))
            #print(force)
            print(trainavgloss)

            if trainloss > 100000:
                break
            losses.append(trainavgloss)

            size = 0
            magnitude = 0
            iteration = 0
            j = trainsize



    print(losses)
    print(test_losses)

    torch.save(seqmodel.state_dict(), PATH)


