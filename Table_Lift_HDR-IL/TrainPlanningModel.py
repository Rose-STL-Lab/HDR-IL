import random

import utils
from Models import HDRIL_Models
from torch import optim
import torch
import torch.nn as nn

from torch.utils import data
import dgl

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

params = {


    "runs" : 2500,
    "runsize" : 70,
    "load_model" : False,
    "train_model" : True,
    "n_epochs" : 2,
    "sequencelength" : 70,
    "n_hidden" :  21,
    "n_latent" : 512,
    "n_iters" : 20,
    "lr_encoder": .00005,
    "lr_decoder": .00005

}

PATH = 'planningmodel.pt'



trainsize = params["runs"]*params["runsize"]
train_set = utils.BaxterDataset()


input_size = train_set.columns.size()[1]
maxsize = len(train_set)
sequencelength = params["sequencelength"]

trainingdata = data.Subset(train_set, indices=list(range(0, trainsize)))
testdata = data.Subset(train_set, indices=list(range(trainsize, maxsize)))




g2 = dgl.DGLGraph().to(device)
nodes = input_size
g2.add_nodes(nodes)
# A couple edges one-by-one
for i in range(0, nodes):
    for j in range(0, nodes):
        g2.add_edge(i, j)


"""The model"""

input_size = input_size
output_size = input_size
n_latent = params["n_latent"]
n_iters = params["n_iters"]

current_loss = 0
test_loss_total = 0
all_losses = []
test_losses = []
losses = []

force = []


gcn1 = HDRIL_Models.PlanningGAT(g2, int(input_size / nodes), n_latent, input_size, 1)

decoder = HDRIL_Models.PlanningDecoder(input_size, 1, n_latent)

encoder_optimizer = optim.Adam(gcn1.parameters(), lr= params["lr_encoder"])
decoder_optimizer = optim.Adam(decoder.parameters(), lr= params["lr_decoder"])

criterion = nn.CrossEntropyLoss()

seqmodel = HDRIL_Models.PlanningVAE(input_size, output_size, n_latent, gcn1, decoder, encoder_optimizer,
                                    decoder_optimizer,
                                    criterion)

seqmodel = seqmodel.cuda()



"""Model training"""

if (params["load_model"] == True):
    seqmodel.load_state_dict(torch.load(PATH))



if (params["train_model"] == True):



    for epoch_idx in range(params["n_epochs"]):

        print(epoch_idx)

        i = 0

        while i < trainsize:



            s = random.randint(0, params["runs"] - 1) * params["runsize"]
            primitive = train_set.labels[s]


            c, p = train_set[s:s + sequencelength]
            c1, _ = train_set[s + 1:s + 1 + sequencelength]
            i += sequencelength

            rows = int(c.size()[0] / sequencelength)

            startcord = c.reshape(rows, sequencelength, input_size).transpose(0, 1).cuda()
            endcord = c1.reshape(rows, sequencelength, input_size).transpose(0, 1).cuda()
            target = p.reshape(rows, sequencelength, 1).transpose(0, 1).cuda()

            trainloss = 0
            testloss = 0

            avgloss = seqmodel.train(startcord.float(), target.float())

            trainavgloss = avgloss

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







