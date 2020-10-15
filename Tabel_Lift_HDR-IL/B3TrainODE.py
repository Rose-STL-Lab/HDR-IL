
import dgl
import seaborn as sns

import B1DataProcessing
import B2Model

sns.color_palette("bright")

from torch import optim

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


runs = 2500
runsize = 70
load_model = True
train_model = False
evaluate = False

n_epochs = 25




trainsize = runsize*runs
test_size = runsize

train_set = B1DataProcessing.BaxterDataset()


trainingdata = data.Subset(train_set, indices=list(range(0, trainsize)))

testdata = data.Subset(train_set, indices=list(range(trainsize, 10200)))


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



# initialize encoders and decoders



activation = nn.Sequential()


g2 = dgl.DGLGraph()
nodes = 21
totalnodes2 = 21
g2.add_nodes(nodes)
# A couple edges one-by-one
for i in range(0, nodes):
    for j in range(0, nodes):
        g2.add_edge(i, j)


#g.add_edge(0,0)
#g.add_edge(1,1)
#g.add_edge(2,2)



#print("sizes", int(input_size/nodes), input_size, n_latent)

out_size = 100

gcn1 = B2Model.GAT(g2, int(input_size / totalnodes2), n_latent, n_hidden, 1)
gcn2 = B2Model.GAT(g2, int(input_size / totalnodes2), n_latent, n_hidden, 1)
gcn3 = B2Model.GAT2(g2, int(input_size / totalnodes2), n_latent, n_hidden, 1)
gcn4 = B2Model.GAT2(g2, int(input_size / totalnodes2), n_latent, n_hidden, 1)
gcn5 = B2Model.GAT2(g2, int(input_size / totalnodes2), n_latent, n_hidden, 1)
gcn6 = B2Model.GAT2(g2, int(input_size / totalnodes2), n_latent, n_hidden, 1)




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

p1encoder_optimizer = optim.Adam(gcn1.parameters(), lr=.0002)
p1decoder_optimizer = optim.Adam(p1decoder.parameters(), lr=.00004)
p2encoder_optimizer = optim.Adam(gcn2.parameters(), lr=.0002)
p2decoder_optimizer = optim.Adam(p2decoder.parameters(), lr=.00004)
p3encoder_optimizer = optim.Adam(gcn3.parameters(), lr=.0002)
p3decoder_optimizer = optim.Adam(p3decoder.parameters(), lr=.00004)
p4encoder_optimizer = optim.Adam(gcn4.parameters(), lr=.0002)
p4decoder_optimizer = optim.Adam(p4decoder.parameters(), lr=.00004)
p5encoder_optimizer = optim.Adam(gcn5.parameters(), lr=.0002)
p5decoder_optimizer = optim.Adam(p5decoder.parameters(), lr=.00004)
p6encoder_optimizer = optim.Adam(gcn6.parameters(), lr=.0002)
p6decoder_optimizer = optim.Adam(p6decoder.parameters(), lr=.00004)

criterion = nn.MSELoss()
# criterion = nn.BCELoss()
# criterion = nn.CrossEntropyLoss()
# criterion = nn.NLLLoss()
#criterion = dtw.DTW_Loss()
# seqmodel = ODEModel.ODEVAE(input_size, output_size, n_latent, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)


graspmodel = B2Model.ODEVAE(input_size, output_size, n_latent, gcn1, p1decoder, p1encoder_optimizer,
                            p1decoder_optimizer, criterion)
sidemodel = B2Model.ODEVAE(input_size, output_size, n_latent, gcn2, p2decoder, p2encoder_optimizer,
                           p2decoder_optimizer, criterion)
liftmodel = B2Model.ODEVAE(input_size, output_size, n_latent, gcn3, p3decoder, p3encoder_optimizer,
                           p3decoder_optimizer, criterion)
extendmodel = B2Model.ODEVAE(input_size, output_size, n_latent, gcn4, p4decoder, p4encoder_optimizer,
                             p4decoder_optimizer, criterion)
placemodel = B2Model.ODEVAE(input_size, output_size, n_latent, gcn5, p5decoder, p5encoder_optimizer,
                            p5decoder_optimizer, criterion)
retractmodel = B2Model.ODEVAE(input_size, output_size, n_latent, gcn6, p6decoder, p6encoder_optimizer,
                              p6decoder_optimizer, criterion)




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
liftmodel = liftmodel.cuda()
extendmodel = extendmodel.cuda()
placemodel = placemodel.cuda()
retractmodel = retractmodel.cuda()
sidemodel = sidemodel.cuda()

encoder = A2SoftmaxModel.RNNEncoder(input_size, output_size, n_latent)
decoder = A2SoftmaxModel.NeuralODEDecoder(input_size, output_size, n_latent)

# criterion = nn.MSELoss()
# criterion = nn.BCELoss()
#criterion = nn.CrossEntropyLoss()
# criterion = nn.NLLLoss()


"""Model training"""

path1 = 'graspmodel.pt'
path2 = 'liftmodel.pt'
path3 = 'extendmodel.pt'
path4 = 'placemodel.pt'
path5 = 'retractmodel.pt'
path6 = 'sidemodel.pt'



if (load_model == True):
    graspmodel.load_state_dict(torch.load(path1))
    liftmodel.load_state_dict(torch.load(path2))
    extendmodel.load_state_dict(torch.load(path3))
    placemodel.load_state_dict(torch.load(path4))
    retractmodel.load_state_dict(torch.load(path5))
    sidemodel.load_state_dict(torch.load(path6))

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

            if primitive == 0:

                sequencelength = 10

            else:
                sequencelength = 12

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

                avgloss = graspmodel.train(startcord, endcord)
                # print(graspmodel(startcord.float()))

                #magnitude = torch.abs(endcord)
                #trainavgloss = (math.sqrt(avgloss) / (
                            #torch.sum(magnitude) / (endcord.size()[0] * endcord.size()[1]))).tolist()



            elif primitive == 1:
                print("train side")

                avgloss = sidemodel.train(startcord, endcord)
                # print(graspmodel(startcord.float()))

                #magnitude = torch.abs(endcord)
                #trainavgloss = (math.sqrt(avgloss) / (
                #        torch.sum(magnitude) / (endcord.size()[0] * endcord.size()[1]))).tolist()

            elif primitive == 2:
                print("train lift")

                avgloss = liftmodel.train(startcord, endcord)
                # print(graspmodel(startcord.float()))

                #magnitude = torch.abs(endcord)
                #trainavgloss = (math.sqrt(avgloss) / (
                            #torch.sum(magnitude) / (endcord.size()[0] * endcord.size()[1]))).tolist()

            

            elif primitive == 3:
                print("train extend")

                avgloss = extendmodel.train(startcord, endcord)
                # print(graspmodel(startcord.float()))

                #magnitude = torch.abs(endcord)
                #trainavgloss = (math.sqrt(avgloss) / (
                #        torch.sum(magnitude) / (endcord.size()[0] * endcord.size()[1]))).tolist()
            elif primitive == 4:
                print("train place")

                avgloss = placemodel.train(startcord, endcord)
                # print(graspmodel(startcord.float()))

                #magnitude = torch.abs(endcord)
                #trainavgloss = (math.sqrt(avgloss) / (
                #        torch.sum(magnitude) / (endcord.size()[0] * endcord.size()[1]))).tolist()
            elif primitive == 5:
                print("train retract")

                avgloss = retractmodel.train(startcord, endcord)
                # print(graspmodel(startcord.float()))

                #magnitude = torch.abs(endcord)
                #trainavgloss = (math.sqrt(avgloss) / (
                #        torch.sum(magnitude) / (endcord.size()[0] * endcord.size()[1]))).tolist()

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


            """
            j = trainsize
            while j < test_size + trainsize:
                iteration += 1

                primitive = train_set.y[j][-1]

                if primitive == 0:
                    sequencelength = 10

                else:
                    sequencelength = 12

                length = int(sequencelength)
                c, f, y, t = train_set[j:j + sequencelength]
                j += sequencelength
                rows = int(c.size()[0] / length)

                startcord = c.reshape(rows, length, input_size).transpose(0, 1).cuda()
                endcord = f.reshape(rows, length, output_size).transpose(0, 1).cuda()

                force = y.reshape(rows, length, input_size2).transpose(0, 1).cuda()
                time = t.reshape(length, rows, 1).cuda()

                # diff = endcord-startcord
                diff = torch.cat((endcord, startcord), 2)

                if primitive == 0:
                    testloss += graspmodel.evaluate(startcord, endcord)
                    #magnitude += torch.sum(torch.abs(endcord))
                    #size += (((endcord.size()[0] * endcord.size()[1])))

                elif primitive == 1:

                    testloss += liftmodel.evaluate(startcord, endcord)
                    #magnitude += torch.sum(torch.abs(endcord))
                    #size += (((endcord.size()[0] * endcord.size()[1])))


                elif primitive == 2:

                    testloss += extendmodel.evaluate(startcord, endcord)
                    #magnitude += torch.sum(torch.abs(endcord))
                    #size += (((endcord.size()[0] * endcord.size()[1])))

                elif primitive == 3:

                    testloss += placemodel.evaluate(startcord, endcord)
                    #magnitude += torch.sum(torch.abs(endcord))
                    #size += (((endcord.size()[0] * endcord.size()[1])))

                elif y[0][-1] == 4:

                    testloss += retractmodel.evaluate(startcord, endcord)
                    #magnitude += torch.sum(torch.abs(endcord))
                    #size += (((endcord.size()[0] * endcord.size()[1])))
                elif y[0][-1] == 5:

                    testloss += sidemodel.evaluate(startcord, endcord)
                    #magnitude += torch.sum(torch.abs(endcord))
                    #size += (((endcord.size()[0] * endcord.size()[1])))

                else:
                    print("extra", primitive, j)

            #testavgloss = math.sqrt(testloss / iteration) / ((torch.sum(magnitude) / size).tolist())
            testloss = testloss/iteration

            test_losses.append(testloss)

            print(testloss)
            """


    torch.save(graspmodel.state_dict(), path1)
    torch.save(liftmodel.state_dict(), path2)
    torch.save(extendmodel.state_dict(), path3)
    torch.save(placemodel.state_dict(), path4)
    torch.save(retractmodel.state_dict(), path5)
    torch.save(sidemodel.state_dict(), path6)

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




    # diff = b - a

if evaluate == True:
    s = 0

    dtwerrors = []

    while s < runs* (runsize - 1):
        i += 1


        primitive = train_set.y[s][-1]
        print("primitive", primitive)

        if primitive == 0:

            sequencelength = 10

        else:
            sequencelength = 12

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

            avgloss = graspmodel.evaluate(startcord, endcord)
            # print(graspmodel(startcord.float()))

            # magnitude = torch.abs(endcord)
            # trainavgloss = (math.sqrt(avgloss) / (
            # torch.sum(magnitude) / (endcord.size()[0] * endcord.size()[1]))).tolist()



        elif primitive == 1:
            print("train side")

            avgloss = sidemodel.evaluate(startcord, endcord)
            # print(graspmodel(startcord.float()))

            # magnitude = torch.abs(endcord)
            # trainavgloss = (math.sqrt(avgloss) / (
            #        torch.sum(magnitude) / (endcord.size()[0] * endcord.size()[1]))).tolist()

        elif primitive == 2:
            print("train lift")

            avgloss = liftmodel.evaluate(startcord, endcord)
            # print(graspmodel(startcord.float()))

            # magnitude = torch.abs(endcord)
            # trainavgloss = (math.sqrt(avgloss) / (
            # torch.sum(magnitude) / (endcord.size()[0] * endcord.size()[1]))).tolist()



        elif primitive == 3:
            print("train extend")

            avgloss = extendmodel.evaluate(startcord, endcord)
            # print(graspmodel(startcord.float()))

            # magnitude = torch.abs(endcord)
            # trainavgloss = (math.sqrt(avgloss) / (
            #        torch.sum(magnitude) / (endcord.size()[0] * endcord.size()[1]))).tolist()
        elif primitive == 4:
            print("train place")

            avgloss = placemodel.evaluate(startcord, endcord)
            # print(graspmodel(startcord.float()))

            # magnitude = torch.abs(endcord)
            # trainavgloss = (math.sqrt(avgloss) / (
            #        torch.sum(magnitude) / (endcord.size()[0] * endcord.size()[1]))).tolist()
        elif primitive == 5:
            print("train retract")

            avgloss = retractmodel.evaluate(startcord, endcord)
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










