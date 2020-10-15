import math

import dgl
import seaborn as sns

import B1DataProcessing
import B2ODEModel

sns.color_palette("bright")

from torch import optim

import csv
import torch
import torch.nn as nn
import random

from torch.utils import data

use_cuda = torch.cuda.is_available()

"""Read in the files"""
"""Options
runs - number of runs
runsize - the size of each demonstration
load_model - load trained parameters
train_model - train the model
evaluate - calculate dtw errors
epochs - number of epochs to run for training. Each epoch is the number of runs*runsize

"""

runs = 2500
runsize = 70
load_model = True
train_model = False
evaluate = False

n_epochs = 5

trainsize = runsize * runs
test_size = runsize

train_set = B1DataProcessing.BaxterDataset()

input_size = train_set.c.size()[1]
output_size = train_set.f.size()[1]
input_size2 = train_set.y.size()[1]
p1output = train_set.f.size()[1]
p2output = train_set.f.size()[1]
p3output = train_set.f.size()[1]
p4output = train_set.f.size()[1]

"""The model"""
n_hidden = 21
n_latent = 512

current_loss = 0
test_loss_total = 0
all_losses = []
test_losses = []
losses = []

force = []

n_epochs2 = 1

activation = nn.Sequential()

g2 = dgl.DGLGraph()
nodes = 21
totalnodes2 = 21
g2.add_nodes(nodes)
# A couple edges one-by-one
for i in range(0, nodes):
    for j in range(0, nodes):
        g2.add_edge(i, j)

# g.add_edge(0,0)
# g.add_edge(1,1)
# g.add_edge(2,2)


print("sizes", int(input_size / nodes), input_size, n_latent)

out_size = 100

gcn1 = B2ODEModel.GAT(g2, int(input_size / totalnodes2), n_latent, n_hidden, 1)

p1encoder = B2ODEModel.RNNEncoder(input_size, output_size, n_latent)
p1decoder = B2ODEModel.NeuralODEDecoder(input_size, output_size, n_latent)

p1encoder_optimizer = optim.Adam(gcn1.parameters(), lr=.00004)
p1decoder_optimizer = optim.Adam(p1decoder.parameters(), lr=.00004)

criterion = nn.MSELoss()
# criterion = nn.BCELoss()
# criterion = nn.CrossEntropyLoss()
# criterion = nn.NLLLoss()
# criterion = dtw.DTW_Loss()
# seqmodel = ODEModel.ODEVAE(input_size, output_size, n_latent, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)


graspmodel = B2ODEModel.ODEVAE(input_size, output_size, n_latent, gcn1, p1decoder, p1encoder_optimizer,
                               p1decoder_optimizer, criterion)

outputsize = torch.zeros([9, 1, train_set.c.size()[0]])
outputsize2 = torch.zeros([12, 1, train_set.c.size()[0]])

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

graspmodel = graspmodel.cuda()

# criterion = nn.MSELoss()
# criterion = nn.BCELoss()
# criterion = nn.CrossEntropyLoss()
# criterion = nn.NLLLoss()


"""Model training"""

path1 = 'graspmodel.pt'

if (load_model == True):
    graspmodel.load_state_dict(torch.load(path1))

    # rint(graspmodel.state_dict())

    # print(graspmodel.state_dict().keys())
    # print(graspmodel.state_dict()['encoder.layer1.heads.0.fc.weight'])

if (train_model == True):
    for epoch_idx in range(n_epochs):

        s = random.randint(0, runs - 1) * runsize

        for i in range(0, runs):
            i += 1

            if s % runsize == 0:
                s = random.randint(0, runs - 1) * runsize
            print(s)

            # primitive = train_set.y[s][-1]
            # print("primitive", primitive)

            # if primitive == 0:

            #    sequencelength = 10

            # else:
            #    sequencelength = 12

            sequencelength = 70

            length = int(sequencelength)
            c, f, y, t = train_set[s:s + sequencelength]

            s += sequencelength
            rows = int(c.size()[0] / length)
            features1 = int(input_size / totalnodes2)
            output1 = int(output_size / totalnodes2)

            startcord = c.reshape(rows, length, input_size).transpose(0, 1).cuda()
            endcord = f.reshape(rows, length, output_size).transpose(0, 1).cuda()
            force = y.reshape(rows, length, input_size2).transpose(0, 1).cuda()
            # p1 = p1.reshape(rows, length, p1output).transpose(0, 1).cuda()
            # p2 = p2.reshape(rows, length, p2output).transpose(0, 1).cuda()
            # p3 = p3.reshape(rows, length, p3output).transpose(0, 1).cuda()
            # p4 = p4.reshape(rows, length, p4output).transpose(0, 1).cuda()

            # print("y", y[0][-1])
            target = t.reshape(length, rows, 1).cuda()

            trainloss = 0
            testloss = 0

            avgloss = graspmodel.train(startcord, endcord)
            # print(graspmodel(startcord.float()))

            # magnitude = torch.abs(endcord)
            # trainavgloss = (math.sqrt(avgloss) / (
            # torch.sum(magnitude) / (endcord.size()[0] * endcord.size()[1]))).tolist()

            # trainavgloss = avgloss

            # print(force)
            print(avgloss)

            if trainloss > 100000:
                break
            losses.append(avgloss)

            size = 0
            magnitude = 0
            iteration = 0

    torch.save(graspmodel.state_dict(), path1)

    print(losses)
    print(test_losses)

    # outputting results

    csvlist = []

    a, b, c, d = train_set[2500:(2500 + sequencelength)]
    a = a.unsqueeze(1).cuda()
    b = b.unsqueeze(1).cuda()
    c = c.unsqueeze(1).cuda()
    d = d.unsqueeze(1).cuda()

    # diff = b - a

    headers = [

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
        'force_1',
        'force_2',
        'force_3',
        'force_4',
        'force_5',
        'force_6',
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
        'right_front_corner_z2',
        'predicted force_1',
        'predicted force_2',
        'predicted force_3',
        'predicted force_4',
        'predicted force_5',
        'predicted force_6'

    ]

    with open('myoutput.csv', 'w', newline='') as f:
        thewriter = csv.writer(f)
        thewriter.writerow(headers)

        for i in range(1, a.size()[0]):
            print(b)
            print(i)
            # thewriter.writerow(a[i][0].tolist() + b[i][0].tolist() + c[i][0].tolist() + graspmodel(a, b)[i][0].tolist())

        thewriter.writerow(losses)
        thewriter.writerow(test_losses)

    import matplotlib.pyplot as plt

    trainingplt, = plt.plot(losses, label='Training data')
    # testplt1, = plt.plot(percenterror[0], label = 'Test data')
    # testplt2, = plt.plot(percenterror[1], label = 'Test data')
    # testplt3, = plt.plot(percenterror[2], label = 'Test data')
    # testplt4, = plt.plot(percenterror[3], label = 'Test data')
    # testplt5, = plt.plot(percenterror[4], label = 'Test data')
    # testplt5, = plt.plot(percenterror[5], label = 'Test data')

    testplt, = plt.plot(test_losses, label='Test data')

    plt.legend(handles=[trainingplt, testplt])
    plt.title('No-Noise Data')

    plt.xlabel('Iteration')
    plt.ylabel('Average Binary Cross Entropy')
    plt.show()

    # outputting results

    # diff = b - a

if evaluate == True:
    s = 0

    dtwerrors = []

    while s < runs * (runsize - 1):
        i += 1

        primitive = train_set.y[s][-1]
        print("primitive", primitive)

        sequencelength = 70

        length = int(sequencelength)
        c, f, y, t = train_set[s:s + sequencelength]

        s += sequencelength
        rows = int(c.size()[0] / length)
        features1 = int(input_size / totalnodes2)
        output1 = int(output_size / totalnodes2)

        startcord = c.reshape(rows, length, input_size).transpose(0, 1).cuda()
        endcord = f.reshape(rows, length, output_size).transpose(0, 1).cuda()
        force = y.reshape(rows, length, input_size2).transpose(0, 1).cuda()
        # p1 = p1.reshape(rows, length, p1output).transpose(0, 1).cuda()
        # p2 = p2.reshape(rows, length, p2output).transpose(0, 1).cuda()
        # p3 = p3.reshape(rows, length, p3output).transpose(0, 1).cuda()
        # p4 = p4.reshape(rows, length, p4output).transpose(0, 1).cuda()

        # print("y", y[0][-1])
        target = t.reshape(length, rows, 1).cuda()

        trainloss = 0
        testloss = 0

        print(primitive)

        avgloss = graspmodel.evaluate(startcord, endcord)
        # print(graspmodel(startcord.float()))

        # trainavgloss = avgloss

        # print(force)
        print(avgloss)

        if trainloss > 100000:
            break
        dtwerrors.append(avgloss)

        size = 0
        magnitude = 0
        iteration = 0

    import numpy as np

    print(np.mean(dtwerrors), "dtw errors")











