import random

import utils
from Models import B2Model



from torch import optim

import torch
import torch.nn as nn

from torch.utils import data


import dgl

use_cuda = torch.cuda.is_available()


"""Options
runs - number of runs
runsize - the size of each demonstration
load_model - load trained parameters
train_model - train the model
epochs - number of epochs to run for training. Each epoch is the number of runs*runsize

"""

params = {


    "runs" : 2500,
    "runsize" : 70,
    "load_model" : False,
    "train_model" : False,
    "n_epochs" : 1,
    "sequencelength" : 70,
    "n_hidden" :  21,
    "n_latent" : 512,
    "n_iters" : 20,
    "lr_encoder": .00005,
    "lr_decoder": .00005

}


trainsize = params["runs"]*params["runsize"]
train_set = utils.BaxterDataset()
maxsize = len(train_set)
sequencelength = params["sequencelength"]

trainingdata = data.Subset(train_set, indices=list(range(0, trainsize)))
testdata = data.Subset(train_set, indices=list(range(trainsize, maxsize)))


input_size = train_set.columns.size()[1]
nodes = input_size

g2 = dgl.DGLGraph()
g2.add_nodes(nodes)
# A couple edges one-by-one
for i in range(0, nodes):
    for j in range(0, nodes):
        g2.add_edge(i, j)


"""The model"""

output_size = input_size
n_latent = params["n_latent"]
n_iters = params["n_iters"]

current_loss = 0
test_loss_total = 0
all_losses = []
test_losses = []
losses = []

force = []





activation = nn.Sequential()



gcn1 = B2Model.GAT2(g2, int(input_size / nodes), n_latent, input_size, 1)

decoder = B2Model.PlanningDecoder(input_size, output_size, n_latent)

encoder_optimizer = optim.Adam(gcn1.parameters(), lr= params["lr_encoder"])
decoder_optimizer = optim.Adam(decoder.parameters(), lr= params["lr_decoder"])

criterion = nn.CrossEntropyLoss()

seqmodel = B2Model.VAE(input_size, output_size, n_latent, gcn1, decoder, encoder_optimizer,
                                 decoder_optimizer,
                                 criterion)

seqmodel = seqmodel.cuda()

"""Model training"""

PATH = 'model2stepnonoise.pt'



if (params["train_model"] == True):
    seqmodel.load_state_dict(torch.load(PATH))

if (params["train_model"] == True):

    for epoch_idx in range(params["n_epochs"]):

        print(epoch_idx)

        i = 0

        while i < trainsize:

            primitive = train_set.f[-1][-1]

            s = random.randint(0, params["runs"] - 1) * params["runsize"]

            #if primitive == 0:
            #    sequencelength = 10

            #else:
            #    sequencelength = 12

            length = sequencelength

            c = train_set[s:s + sequencelength]
            c1 = train_set[s + 1:s + 1 + sequencelength]
            i += sequencelength

            rows = int(c.size()[0] / length)

            startcord = c.reshape(rows, length, input_size).transpose(0, 1).cuda()
            endcord = c1.reshape(rows, length, input_size).transpose(0, 1).cuda()

            force = p.reshape(rows, length, output_size).transpose(0, 1).cuda()

            trainloss = 0
            testloss = 0


            avgloss = seqmodel.train(startcord.float(), force.float())

            trainavgloss = avgloss

            print("comparison", seqmodel.getPrimitive(startcord.float(), force))
            print(trainavgloss)

            if trainloss > 100000:
                break
            losses.append(trainavgloss)

            size = 0

            iteration = 0
            j = trainsize

    print(losses)
    print(test_losses)

    torch.save(seqmodel.state_dict(), PATH)







