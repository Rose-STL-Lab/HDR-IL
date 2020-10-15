import math

import dgl
import seaborn as sns

import B1DataProcessing
import B2Model

sns.color_palette("bright")

from torch import optim

import csv
import torch
import torch.nn as nn
import A2SoftmaxModel
import random
import numpy as np



from torch.utils import data

use_cuda = torch.cuda.is_available()

"""Options
runs - number of runs
runsize - the size of each demonstration
load_model - load trained parameters
train_model - train the model
evaluate - calculate dtw errors
epochs - number of epochs to run for training. Each epoch is the number of runs*runsize

"""


runs = 4700
runsize = 130
load_model = True
train_model = False
evaluate = False

n_epochs = 10

trainsize = runsize*runs
test_size = runsize

train_set = B1DataProcessing.BaxterDataset()


trainingdata = data.Subset(train_set, indices=list(range(0, trainsize)))

testdata = data.Subset(train_set, indices=list(range(trainsize, 10200)))

train_loader = torch.utils.data.DataLoader(trainingdata, shuffle=False, batch_size=sequencelength)
test_loader = torch.utils.data.DataLoader(testdata, shuffle=False, batch_size=sequencelength)

input_size = train_set.c.size()[1]
output_size = train_set.f.size()[1]
input_size2 = train_set.y.size()[1]
p1output = train_set.f.size()[1]
p2output = train_set.f.size()[1]
p3output = train_set.f.size()[1]
p4output = train_set.f.size()[1]

"""The model"""
n_hidden = 28
n_latent = 512


current_loss = 0
test_loss_total = 0
all_losses = []
test_losses = []
losses = []

force = []


# initialize encoders and decoders

#encoder = A2SoftmaxModel.RNNEncoder(input_size, output_size, n_latent)
#decoder = A2SoftmaxModel.NeuralODEDecoder(input_size, output_size, n_latent)



activation = nn.Sequential()




g2 = dgl.DGLGraph()
nodes = 28
totalnodes2 = 28
g2.add_nodes(nodes)
# A couple edges one-by-one
for i in range(0, 14):
    for j in range(0, 14):
        g2.add_edge(i, j)


for i in range(14, 28):
    for j in range(0, 14):
        g2.add_edge(i, j)




#for i in range(14, 28):
#    for j in range(0, 14):
#        g2.add_edge(i, j)


#g.add_edge(0,0)
#g.add_edge(1,1)
#g.add_edge(2,2)



print("sizes", int(input_size/nodes), input_size, n_latent)

out_size = 100

gcn1 = B2Model.GAT3(g2, int(input_size / totalnodes2), n_latent, n_hidden, 1)
gcn2 = B2Model.GAT3(g2, int(input_size / totalnodes2), n_latent, n_hidden, 1)
gcn3 = B2Model.GAT3(g2, int(input_size / totalnodes2), n_latent, n_hidden, 1)
gcn4 = B2Model.GAT3(g2, int(input_size / totalnodes2), n_latent, n_hidden, 1)
gcn5 = B2Model.GAT3(g2, int(input_size / totalnodes2), n_latent, n_hidden, 1)
gcn6 = B2Model.GAT3(g2, int(input_size / totalnodes2), n_latent, n_hidden, 1)
gcn7 = B2Model.GAT3(g2, int(input_size / totalnodes2), n_latent, n_hidden, 1)
gcn8 = B2Model.GAT3(g2, int(input_size / totalnodes2), n_latent, n_hidden, 1)
gcn9 = B2Model.GAT3(g2, int(input_size / totalnodes2), n_latent, n_hidden, 1)
gcn10 = B2Model.GAT3(g2, int(input_size / totalnodes2), n_latent, n_hidden, 1)
gcn11 = B2Model.GAT3(g2, int(input_size / totalnodes2), n_latent, n_hidden, 1)
gcn12 = B2Model.GAT3(g2, int(input_size / totalnodes2), n_latent, n_hidden, 1)
gcn13 = B2Model.GAT3(g2, int(input_size / totalnodes2), n_latent, n_hidden, 1)




p1encoder = B2Model.RNNEncoder(input_size, output_size, n_latent)
p1decoder = B2Model.NeuralODEDecoder(input_size, output_size, n_latent)
p2encoder = B2Model.RNNEncoder(input_size, output_size, n_latent)
p2decoder = B2Model.NeuralODEDecoder(input_size, output_size, n_latent)
p3encoder = B2Model.RNNEncoder(input_size, output_size, n_latent)
p3decoder = B2Model.NeuralODEDecoder(input_size, output_size, n_latent)
p4encoder = B2Model.RNNEncoder(input_size, output_size, n_latent)
p4decoder = B2Model.NeuralODEDecoder(input_size, output_size, n_latent)
p5encoder = B2Model.RNNEncoder(input_size, output_size, n_latent)
p5decoder = B2Model.NeuralODEDecoder(input_size, output_size, n_latent)
p6encoder = B2Model.RNNEncoder(input_size, output_size, n_latent)
p6decoder = B2Model.NeuralODEDecoder(input_size, output_size, n_latent)
p7encoder = B2Model.RNNEncoder(input_size, output_size, n_latent)
p7decoder = B2Model.NeuralODEDecoder(input_size, output_size, n_latent)
p8encoder = B2Model.RNNEncoder(input_size, output_size, n_latent)
p8decoder = B2Model.NeuralODEDecoder(input_size, output_size, n_latent)
p9encoder = B2Model.RNNEncoder(input_size, output_size, n_latent)
p9decoder = B2Model.NeuralODEDecoder(input_size, output_size, n_latent)
p10encoder = B2Model.RNNEncoder(input_size, output_size, n_latent)
p10decoder = B2Model.NeuralODEDecoder(input_size, output_size, n_latent)
p11encoder = B2Model.RNNEncoder(input_size, output_size, n_latent)
p11decoder = B2Model.NeuralODEDecoder(input_size, output_size, n_latent)
p12encoder = B2Model.RNNEncoder(input_size, output_size, n_latent)
p12decoder = B2Model.NeuralODEDecoder(input_size, output_size, n_latent)
p13encoder = B2Model.RNNEncoder(input_size, output_size, n_latent)
p13decoder = B2Model.NeuralODEDecoder(input_size, output_size, n_latent)

lr = 0.00005
p1encoder_optimizer = optim.Adam(gcn1.parameters(), lr= lr)
p1decoder_optimizer = optim.Adam(p1decoder.parameters(), lr= lr)
p2encoder_optimizer = optim.Adam(gcn2.parameters(), lr=lr)
p2decoder_optimizer = optim.Adam(p2decoder.parameters(), lr=lr)
p3encoder_optimizer = optim.Adam(gcn3.parameters(), lr=lr)
p3decoder_optimizer = optim.Adam(p3decoder.parameters(), lr=lr)
p4encoder_optimizer = optim.Adam(gcn4.parameters(), lr=lr)
p4decoder_optimizer = optim.Adam(p4decoder.parameters(), lr=lr)
p5encoder_optimizer = optim.Adam(gcn5.parameters(), lr=lr)
p5decoder_optimizer = optim.Adam(p5decoder.parameters(), lr=lr)
p6encoder_optimizer = optim.Adam(gcn6.parameters(), lr=lr)
p6decoder_optimizer = optim.Adam(p6decoder.parameters(), lr=lr)
p7encoder_optimizer = optim.Adam(gcn7.parameters(), lr=lr)
p7decoder_optimizer = optim.Adam(p7decoder.parameters(), lr=lr)
p8encoder_optimizer = optim.Adam(gcn8.parameters(), lr=lr)
p8decoder_optimizer = optim.Adam(p8decoder.parameters(), lr=lr)
p9encoder_optimizer = optim.Adam(gcn9.parameters(), lr=lr)
p9decoder_optimizer = optim.Adam(p9decoder.parameters(), lr=lr)
p10encoder_optimizer = optim.Adam(gcn10.parameters(), lr=lr)
p10decoder_optimizer = optim.Adam(p10decoder.parameters(), lr=lr)
p11encoder_optimizer = optim.Adam(gcn11.parameters(), lr=lr)
p11decoder_optimizer = optim.Adam(p11decoder.parameters(), lr=lr)
p12encoder_optimizer = optim.Adam(gcn12.parameters(), lr=lr)
p12decoder_optimizer = optim.Adam(p12decoder.parameters(), lr=lr)
p13encoder_optimizer = optim.Adam(gcn13.parameters(), lr=lr)
p13decoder_optimizer = optim.Adam(p13decoder.parameters(), lr=lr)

criterion = nn.MSELoss()
# criterion = nn.BCELoss()
# criterion = nn.CrossEntropyLoss()
# criterion = nn.NLLLoss()
#criterion = dtw.DTW_Loss()
# seqmodel = ODEModel.ODEVAE(input_size, output_size, n_latent, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)


model1 = B2Model.ODEVAE(input_size, output_size, n_latent, gcn1, p1decoder, p1encoder_optimizer,
                        p1decoder_optimizer, criterion)
model2 = B2Model.ODEVAE(input_size, output_size, n_latent, gcn2, p2decoder, p2encoder_optimizer,
                        p2decoder_optimizer, criterion)
model3 = B2Model.ODEVAE(input_size, output_size, n_latent, gcn3, p3decoder, p3encoder_optimizer,
                        p3decoder_optimizer, criterion)
model4 = B2Model.ODEVAE(input_size, output_size, n_latent, gcn4, p4decoder, p4encoder_optimizer,
                        p4decoder_optimizer, criterion)
model5 = B2Model.ODEVAE(input_size, output_size, n_latent, gcn5, p5decoder, p5encoder_optimizer,
                        p5decoder_optimizer, criterion)
model6 = B2Model.ODEVAE(input_size, output_size, n_latent, gcn6, p6decoder, p6encoder_optimizer,
                        p6decoder_optimizer, criterion)
model7 = B2Model.ODEVAE(input_size, output_size, n_latent, gcn7, p7decoder, p7encoder_optimizer,
                        p7decoder_optimizer, criterion)
model8 = B2Model.ODEVAE(input_size, output_size, n_latent, gcn8, p8decoder, p8encoder_optimizer,
                        p8decoder_optimizer, criterion)
model9 = B2Model.ODEVAE(input_size, output_size, n_latent, gcn9, p9decoder, p9encoder_optimizer,
                        p9decoder_optimizer, criterion)
model10 = B2Model.ODEVAE(input_size, output_size, n_latent, gcn10, p10decoder, p10encoder_optimizer,
                         p10decoder_optimizer, criterion)
model11 = B2Model.ODEVAE(input_size, output_size, n_latent, gcn11, p11decoder, p11encoder_optimizer,
                         p11decoder_optimizer, criterion)
model12 = B2Model.ODEVAE(input_size, output_size, n_latent, gcn12, p12decoder, p12encoder_optimizer,
                         p12decoder_optimizer, criterion)
model13 = B2Model.ODEVAE(input_size, output_size, n_latent, gcn13, p13decoder, p13encoder_optimizer,
                         p13decoder_optimizer, criterion)




outputsize = torch.zeros([10, 1, train_set.c.size()[0]])
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


model1 = model1.cuda()
model2 = model2.cuda()
model3 = model3.cuda()
model4 = model4.cuda()
model5 = model5.cuda()
model6 = model6.cuda()
model7 = model7.cuda()
model8 = model8.cuda()
model9 = model9.cuda()
model10 = model10.cuda()
model11 = model11.cuda()
model12 = model12.cuda()
model13 = model13.cuda()

encoder = A2SoftmaxModel.RNNEncoder(input_size, output_size, n_latent)
decoder = A2SoftmaxModel.NeuralODEDecoder(input_size, output_size, n_latent)

encoder_optimizer = optim.Adam(encoder.parameters(), lr=.001)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=.001)
# criterion = nn.MSELoss()
# criterion = nn.BCELoss()
#criterion = nn.CrossEntropyLoss()
# criterion = nn.NLLLoss()


"""Model training"""

path1 = 'model1.pt'
path2 = 'model2.pt'
path3 = 'model3.pt'
path4 = 'model4.pt'
path5 = 'model5.pt'
path6 = 'model6.pt'
path7 = 'model7.pt'
path8 = 'model8.pt'
path9 = 'model9.pt'
path10 = 'model10.pt'
path11 = 'model11.pt'
path12 = 'model12.pt'
path13 = 'model13.pt'


load_model = True
train_model = False
train_auto = False
evaluate = True

if (load_model == True):
    model1.load_state_dict(torch.load(path1))
    model2.load_state_dict(torch.load(path2))
    model3.load_state_dict(torch.load(path3))
    model4.load_state_dict(torch.load(path4))
    model5.load_state_dict(torch.load(path5))
    model6.load_state_dict(torch.load(path6))
    model7.load_state_dict(torch.load(path7))
    model8.load_state_dict(torch.load(path8))
    model9.load_state_dict(torch.load(path9))
    model10.load_state_dict(torch.load(path10))
    model11.load_state_dict(torch.load(path11))
    model12.load_state_dict(torch.load(path12))
    model13.load_state_dict(torch.load(path13))


    #rint(graspmodel.state_dict())


    #print(graspmodel.state_dict().keys())
    #print(graspmodel.state_dict()['encoder.layer1.heads.0.fc.weight'])

if (train_model == True):
    for epoch_idx in range(n_epochs):

        s = random.randint(0, runs-1) * runsize

        for i in range(0, runs):
            i += 1

            if s % runsize == 0:
                s = random.randint(0, runs-1) * runsize
            print(s)

            primitive = train_set.y[s][-1]
            print("primitive", primitive)



            sequencelength = 10


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
            if primitive == 0:
                print("train grasp")

                avgloss = model1.train(startcord, endcord)
                # print(graspmodel(startcord.float()))

                #magnitude = torch.abs(endcord)
                #trainavgloss = (math.sqrt(avgloss) / (
                            #torch.sum(magnitude) / (endcord.size()[0] * endcord.size()[1]))).tolist()



            elif primitive == 1:
                print("train connect")

                avgloss = model2.train(startcord, endcord)
                # print(graspmodel(startcord.float()))

                #magnitude = torch.abs(endcord)
                #trainavgloss = (math.sqrt(avgloss) / (
                #        torch.sum(magnitude) / (endcord.size()[0] * endcord.size()[1]))).tolist()


            elif primitive == 2:
                print("train lift")

                avgloss = model3.train(startcord, endcord)
                # print(graspmodel(startcord.float()))

                #magnitude = torch.abs(endcord)
                #trainavgloss = (math.sqrt(avgloss) / (
                            #torch.sum(magnitude) / (endcord.size()[0] * endcord.size()[1]))).tolist()
            elif primitive == 3:
                print("train 3")

                avgloss = model4.train(startcord, endcord)
                # print(graspmodel(startcord.float()))

                #magnitude = torch.abs(endcord)
                #trainavgloss = (math.sqrt(avgloss) / (
                            #torch.sum(magnitude) / (endcord.size()[0] * endcord.size()[1]))).tolist()
            elif primitive == 4:
                print("train 4")

                avgloss = model5.train(startcord, endcord)
                # print(graspmodel(startcord.float()))

                #magnitude = torch.abs(endcord)
                #trainavgloss = (math.sqrt(avgloss) / (
                            #torch.sum(magnitude) / (endcord.size()[0] * endcord.size()[1]))).tolist()
            elif primitive == 5:
                print("train lift")

                avgloss = model6.train(startcord, endcord)
                # print(graspmodel(startcord.float()))

                #magnitude = torch.abs(endcord)
                #trainavgloss = (math.sqrt(avgloss) / (
                            #torch.sum(magnitude) / (endcord.size()[0] * endcord.size()[1]))).tolist()
            elif primitive == 6:
                print("train lift")

                avgloss = model7.train(startcord, endcord)
                # print(graspmodel(startcord.float()))

                #magnitude = torch.abs(endcord)
                #trainavgloss = (math.sqrt(avgloss) / (
                            #torch.sum(magnitude) / (endcord.size()[0] * endcord.size()[1]))).tolist()

            elif primitive == 7:
                print("train 7")

                avgloss = model8.train(startcord, endcord)
                # print(graspmodel(startcord.float()))

                #magnitude = torch.abs(endcord)
                #trainavgloss = (math.sqrt(avgloss) / (
                            #torch.sum(magnitude) / (endcord.size()[0] * endcord.size()[1]))).tolist()

            elif primitive == 8:
                print("train 8")

                avgloss = model9.train(startcord, endcord)
                # print(graspmodel(startcord.float()))

                #magnitude = torch.abs(endcord)
                #trainavgloss = (math.sqrt(avgloss) / (
                            #torch.sum(magnitude) / (endcord.size()[0] * endcord.size()[1]))).tolist()

            elif primitive == 9 :
                print("train 9")

                avgloss = model10.train(startcord, endcord)
                # print(graspmodel(startcord.float()))

                #magnitude = torch.abs(endcord)
                #trainavgloss = (math.sqrt(avgloss) / (
                            #torch.sum(magnitude) / (endcord.size()[0] * endcord.size()[1]))).tolist()

            elif primitive == 10 :
                print("train 10")

                avgloss = model11.train(startcord, endcord)
                # print(graspmodel(startcord.float()))

                #magnitude = torch.abs(endcord)
                #trainavgloss = (math.sqrt(avgloss) / (
                            #torch.sum(magnitude) / (endcord.size()[0] * endcord.size()[1]))).tolist()

            elif primitive == 11 :
                print("train 11")

                avgloss = model12.train(startcord, endcord)
                # print(graspmodel(startcord.float()))

                #magnitude = torch.abs(endcord)
                #trainavgloss = (math.sqrt(avgloss) / (
                            #torch.sum(magnitude) / (endcord.size()[0] * endcord.size()[1]))).tolist()

            elif primitive == 12 :
                print("train 12")

                avgloss = model13.train(startcord, endcord)
                # print(graspmodel(startcord.float()))

                #magnitude = torch.abs(endcord)
                #trainavgloss = (math.sqrt(avgloss) / (
                            #torch.sum(magnitude) / (endcord.size()[0] * endcord.size()[1]))).tolist()

            



            else:
                print("train none")




            # trainavgloss = avgloss

            # print(force)
            print(avgloss)
            


            if trainloss > 100000:
                break
            losses.append(avgloss)

            size = 0
            magnitude = 0
            iteration = 0



    torch.save(model1.state_dict(), path1)
    torch.save(model2.state_dict(), path2)
    torch.save(model3.state_dict(), path3)
    torch.save(model4.state_dict(), path4)
    torch.save(model5.state_dict(), path5)
    torch.save(model6.state_dict(), path6)
    torch.save(model7.state_dict(), path7)
    torch.save(model8.state_dict(), path8)
    torch.save(model9.state_dict(), path9)
    torch.save(model10.state_dict(), path10)
    torch.save(model11.state_dict(), path11)
    torch.save(model12.state_dict(), path12)
    torch.save(model13.state_dict(), path13)

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
            #thewriter.writerow(a[i][0].tolist() + b[i][0].tolist() + c[i][0].tolist() + graspmodel(a, b)[i][0].tolist())

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










    # diff = b - a



if evaluate == True:
    s = 0

    dtwerrors = []

    while s < runs* (runsize - 1):
        i += 1


        primitive = train_set.y[s][-1]
        print("primitive", primitive)



        sequencelength = 10



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
        if primitive == 0:
            print("train grasp")

            avgloss = model1.evaluate(startcord, endcord)
            # print(graspmodel(startcord.float()))

            # magnitude = torch.abs(endcord)
            # trainavgloss = (math.sqrt(avgloss) / (
            # torch.sum(magnitude) / (endcord.size()[0] * endcord.size()[1]))).tolist()



        elif primitive == 1:
            print("train side")

            avgloss = model2.evaluate(startcord, endcord)
            # print(graspmodel(startcord.float()))

            # magnitude = torch.abs(endcord)
            # trainavgloss = (math.sqrt(avgloss) / (
            #        torch.sum(magnitude) / (endcord.size()[0] * endcord.size()[1]))).tolist()

        elif primitive == 2:
            print("train lift")

            avgloss = model3.evaluate(startcord, endcord)
            # print(graspmodel(startcord.float()))

            # magnitude = torch.abs(endcord)
            # trainavgloss = (math.sqrt(avgloss) / (
            # torch.sum(magnitude) / (endcord.size()[0] * endcord.size()[1]))).tolist()



        elif primitive == 3:
            print("train extend")

            avgloss = model4.evaluate(startcord, endcord)
            # print(graspmodel(startcord.float()))

            # magnitude = torch.abs(endcord)
            # trainavgloss = (math.sqrt(avgloss) / (
            #        torch.sum(magnitude) / (endcord.size()[0] * endcord.size()[1]))).tolist()
        elif primitive == 4:
            print("train place")

            avgloss = model5.evaluate(startcord, endcord)
            # print(graspmodel(startcord.float()))

            # magnitude = torch.abs(endcord)
            # trainavgloss = (math.sqrt(avgloss) / (
            #        torch.sum(magnitude) / (endcord.size()[0] * endcord.size()[1]))).tolist()
        elif primitive == 5:
            print("train retract")

            avgloss = model6.evaluate(startcord, endcord)
            # print(graspmodel(startcord.float()))

            # magnitude = torch.abs(endcord)
            # trainavgloss = (math.sqrt(avgloss) / (
            #        torch.sum(magnitude) / (endcord.size()[0] * endcord.size()[1]))).tolist()

        elif primitive == 6:
            print("train retract")

            avgloss = model7.evaluate(startcord, endcord)
            # print(graspmodel(startcord.float()))

            # magnitude = torch.abs(endcord)
            # trainavgloss = (math.sqrt(avgloss) / (
            #        torch.sum(magnitude) / (endcord.size()[0] * endcord.size()[1]))).tolist()

        elif primitive == 7:
            print("train retract")

            avgloss = model8.evaluate(startcord, endcord)
            # print(graspmodel(startcord.float()))

            # magnitude = torch.abs(endcord)
            # trainavgloss = (math.sqrt(avgloss) / (
            #        torch.sum(magnitude) / (endcord.size()[0] * endcord.size()[1]))).tolist()

        elif primitive == 8:
            print("train retract")

            avgloss = model9.evaluate(startcord, endcord)
            # print(graspmodel(startcord.float()))

            # magnitude = torch.abs(endcord)
            # trainavgloss = (math.sqrt(avgloss) / (
            #        torch.sum(magnitude) / (endcord.size()[0] * endcord.size()[1]))).tolist()
        elif primitive == 9:
            print("train retract")

            avgloss = model10.evaluate(startcord, endcord)
            # print(graspmodel(startcord.float()))

            # magnitude = torch.abs(endcord)
            # trainavgloss = (math.sqrt(avgloss) / (
            #        torch.sum(magnitude) / (endcord.size()[0] * endcord.size()[1]))).tolist()

        elif primitive == 10:
            print("train retract")

            avgloss = model11.evaluate(startcord, endcord)
            # print(graspmodel(startcord.float()))

            # magnitude = torch.abs(endcord)
            # trainavgloss = (math.sqrt(avgloss) / (
            #        torch.sum(magnitude) / (endcord.size()[0] * endcord.size()[1]))).tolist()

        elif primitive == 11:
            print("train retract")

            avgloss = model12.evaluate(startcord, endcord)
            # print(graspmodel(startcord.float()))

            # magnitude = torch.abs(endcord)
            # trainavgloss = (math.sqrt(avgloss) / (
            #        torch.sum(magnitude) / (endcord.size()[0] * endcord.size()[1]))).tolist()

        elif primitive == 12:
            print("train retract")

            avgloss = model13.evaluate(startcord, endcord)
            # print(graspmodel(startcord.float()))

            # magnitude = torch.abs(endcord)
            # trainavgloss = (math.sqrt(avgloss) / (
            #        torch.sum(magnitude) / (endcord.size()[0] * endcord.size()[1]))).tolist()

        else:
            print("train none")

        # trainavgloss = avgloss

        # print(force)
        print(avgloss)

        if trainloss > 100000:
            break
        dtwerrors.append(avgloss)

        size = 0
        magnitude = 0
        iteration = 0

    print(np.mean(dtwerrors), "dtw errors")







