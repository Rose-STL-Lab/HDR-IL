import dgl
import utils
from Models import B2Model
from torch import optim
import torch
import torch.nn as nn
import numpy as np
from torch.utils import data



params = {


    "runs": 1,  #number of runs
    "runsize": 54,  #the size of each demonstration
    "load_model": False,   #start from previously saved model
    "train_model" : True,   #train the model
    "evaluate" : False, #calculate the dtw error of the projections
    "n_epochs" : 100, #number of epochs to run for training. Each epoch is the number of runs*runsize
    "n_latent" : 512,
    "lr_encoder": .00005,
    "lr_decoder": .00005

}


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_set = utils.BaxterDataset()


trainsize = params["runsize"] * params["runs"] #61*1000
test_size = len(train_set)

trainingdata = data.Subset(train_set, indices=list(range(0, trainsize)))


input_size = train_set.columns.size()[1]
output_size = input_size


"""Model directories"""

path1 = 'xy/graspmodel.pt'
path2 = 'xy/liftmodel.pt'
path3 = 'xy/extendmodel.pt'
path4 = 'xy/placemodel.pt'
path5 = 'xy/retractmodel.pt'
path6 = 'xy/sidemodel.pt'


"""Initialize the model parameters"""


n_latent = params['n_latent']

current_loss = 0
test_loss_total = 0
all_losses = []
dtw_errors = []
test_losses = []
losses = []

force = []

# initialize encoders and decoders

headers = utils.outputHeaders()


"""Initialize the graph"""
g2 = dgl.DGLGraph().to(device)
nodes = input_size
g2.add_nodes(nodes)
# A couple edges one-by-one
for i in range(0, nodes):
    for j in range(0, nodes):
        g2.add_edge(i, j)



criterion = nn.MSELoss()

gcn1 = B2Model.GAT(g2, int(input_size / nodes), n_latent, output_size, 1)
p1decoder = B2Model.Decoder(input_size, output_size, n_latent)

p1encoder_optimizer = optim.Adam(gcn1.parameters(), lr= params["lr_encoder"])
p1decoder_optimizer = optim.Adam(p1decoder.parameters(), lr= params["lr_decoder"])

graspmodel = B2Model.VAE(input_size, output_size, n_latent, gcn1, p1decoder, p1encoder_optimizer,
                            p1decoder_optimizer, criterion)


graspmodel = graspmodel.cuda()



if (params["load_model"] == True):

    graspmodel.load_state_dict(torch.load(path1))


if (params["train_model"] == True):
    for epoch_idx in range(params["n_epochs"]):

        #s = random.randint(0, runs) * runsize

        s = 0

        for i in range(0, params["runs"]):

            sequencelength = params["runsize"]

            length = int(sequencelength)
            c = train_set[s:s + sequencelength]
            c1 = train_set[s + 1:s + 1 + sequencelength]

            #s += sequencelength
            rows = int(c.size()[0] / length)

            startcord = c.reshape(rows, length, input_size).transpose(0, 1).cuda()
            endcord = c1.reshape(rows, length, input_size).transpose(0, 1).cuda()


            trainloss = 0
            testloss = 0


            avgloss = graspmodel.train(startcord, endcord)


            print("avg loss" , avgloss)

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






if params["evaluate"] == True:
    s = 0

    for i in range(0, params["runs"]):

        sequencelength = params["runsize"]

        length = int(sequencelength)
        c = train_set[s:s + sequencelength]
        c1 = train_set[s + 1:s + 1 + sequencelength]

        # s += sequencelength
        rows = int(c.size()[0] / length)
        features1 = int(input_size / nodes)
        output1 = int(output_size / nodes)

        startcord = c.reshape(rows, length, input_size).transpose(0, 1).cuda()
        endcord = c1.reshape(rows, length, input_size).transpose(0, 1).cuda()

        trainloss = 0
        testloss = 0


        avgloss = graspmodel.evaluate(startcord, endcord)

        # trainavgloss = avgloss

        # print(force)
        print(avgloss)

        if trainloss > 100000:
            break
        dtw_errors.append(avgloss)

        size = 0
        magnitude = 0
        iteration = 0

    print(np.mean(dtw_errors), "dtw errors")













