import math

import inline as inline
import matplotlib
import numpy as np
from IPython.display import clear_output
#from tqdm import tqdm_notebook as tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt


import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torch import Tensor, optim
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()




class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()
        self.alphalist = []

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        #print(nodes, "nodes")
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        #print(alpha.size())
        #print(alpha.size(), "alpha")
        #print(nodes.nodes, "nodes")
        #print(nodes.mailbox['e'])
        #print(alpha.size())
        #print(alpha.flatten())
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def getalpha(self, nodes):
        return F.softmax(nodes.mailbox['e'], dim=1)

    def forward(self, h):
        # equation (1)
        #print("h size", h.size())
        z = self.fc(h)
        #print(z.size())

        self.g.ndata['z'] = z
        # equation (2)
        self.g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        self.g.update_all(self.message_func, self.reduce_func)
        #print("weights", self.attn_fc.weight.data)
        #print(self.attn_fc.weight.data.size())

        return self.g.ndata.pop('h')






class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge

    def forward(self, h):

        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)

            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))

    def getalpha(self):
        for h in self.heads:
            print("here")
            h.getalpha()



class GAT(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, latent_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(g, in_dim, latent_dim, num_heads)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = MultiHeadGATLayer(g, latent_dim * num_heads, latent_dim, 1)

        self.input_dim = in_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.nodes = len(g.nodes())



        self.table2hid = nn.Linear(6, hidden_dim)

        self.gru_size = self.nodes * latent_dim
        self.hid2in = nn.Linear(latent_dim, in_dim)
        self.rnn = nn.GRU(self.gru_size, self.gru_size)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.hid1 = nn.Linear(self.gru_size, hidden_dim)
        self.hid2 = nn.Linear(hidden_dim, hidden_dim)
        self.hid3 = nn.Linear(hidden_dim, hidden_dim)
        self.hid4 = nn.Linear(hidden_dim, hidden_dim)
        self.hid5 = nn.Linear(hidden_dim, hidden_dim)

        self.hid2hid = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h, target):
        #print(h[0, 14:21, :].size())

        temp = h[0]
        # t = h[0, 14:21, :].reshape(1, 7)
        # table = self.table2hid(t).unsqueeze(0)
        #print("table")

        # print(target.size(), )

        hidden = torch.zeros((1, 1, self.gru_size)).cuda()

        for i in range(0, target.size()[0]):
            # if i < h.size()[0]:
            #    temp = h[i, 0:14, :]
            #print("temp", temp.size())

            h1 = self.layer1(temp)
            # h2 = F.elu(h1)

            # h2 = self.layer2(h2)

            # print("rnn input", h2.size())
            # print("h1", h1.size())

            rnn_inp = h1.flatten()
            # print(rnn_inp.size())

            out, hid = self.rnn(rnn_inp.unsqueeze(0).unsqueeze(0), hidden)
            # print("out size", out.size(), hid.size(), rnn_inp.size())

            out = out.reshape(self.nodes, self.latent_dim)
            hid = hid.reshape(self.nodes, self.latent_dim)
            hidden = self.layer2(hid)
            # print(hidden.size())
            hidden = hidden.flatten().unsqueeze(0).unsqueeze(0)
            # hidden = hid
            temp = self.hid2in(out).squeeze(0)
            # print(out.size(), "output size")

        h0 = self.hid1(hidden.float())
        # print(table.size(), h0.size(), torch.cat((h0, table), 2).size())
        h0 = self.hid2(h0)
        h0 = self.hid3(h0)
        h0 = self.hid4(h0)
        h0 = self.hid5(h0)

        # print("h0", h0.size())

        #h0 = self.hid2hid(h0)


        #zmean = h0

        h0 = h0
        #print("h0", h0.size(), self.hidden_dim)



        return out, h0



class RNNEncoder(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim):
        super(RNNEncoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.rnn = nn.GRU(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.hid2out = nn.Linear(hidden_dim, output_dim)
        self.hid2 = nn.Linear(hidden_dim, hidden_dim)
        self.hid3 = nn.Linear(hidden_dim, hidden_dim)
        self.hid4 = nn.Linear(hidden_dim, hidden_dim)

        self.soft = nn.Softmax(2)

    def forward(self, x, time):
        x = x.float().cuda()
        xt = torch.cat((x, time.float().cuda()), dim=-1)

        output, h0 = self.rnn(x)  # Reversed

        #print("output size", self.output_dim)


        output, h0 = self.rnn(x)  # Reversed
        output = self.hid2(output)
        output = self.hid3(output)
        output = self.hid4(output)
        outputHid = self.hid2out(output.float())

        return outputHid, h0

    def getCurrentPrimitive(self, x):


        output, h0 = self.rnn(x)

        outputHid = self.hid2out(h0.float())
        softmax = self.soft(outputHid)


        return np.argmax(softmax[-1][-1].detach().cpu())

class NeuralODEDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(NeuralODEDecoder, self).__init__()
        self.input_dim = input_dim

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        func = NNODEF(hidden_dim, hidden_dim, time_invariant=True)
        self.ode = NeuralODE(func)
        self.gru = nn.GRU(hidden_dim, hidden_dim)

        self.bn1 = nn.BatchNorm1d(self.hidden_dim)

        self.l2h = nn.Linear(hidden_dim, hidden_dim)
        self.l2h2 = nn.Linear(hidden_dim, hidden_dim)
        self.l2h3 = nn.Linear(hidden_dim, hidden_dim)
        self.l2h4 = nn.Linear(hidden_dim, hidden_dim)

        self.l2h5 = nn.Linear(hidden_dim, output_dim)
        self.soft = nn.Softmax(2)

    def forward(self, hidden, target):
        # z0 = z0.unsqueeze(1).unsqueeze(1).float().view(1,1,self.output_dim).cuda()


        #print("decoder forward hidden", hidden.size())

        input = torch.zeros(1, 1, hidden.size()[2]).cuda()
        output = torch.zeros(1, target.size()[1], target.size()[2]).cuda()

        for i in range(0, target.size()[0]):
            zs, h = self.gru(input, hidden)
            # print("decoder forward hidden", hidden.size(), time.size(), zs.size())
            # print(hidden)
            # print(time)
            # print(zs)

            input = zs

            zs = self.l2h(zs)
            zs = self.l2h2(zs)
            zs = self.l2h3(zs)
            zs = self.l2h4(zs)
            hs = self.l2h5(zs)

            hidden = h

            #print("hs", hs.size())

            softmax = self.soft(hs)

            output = torch.cat((output, softmax))

        # print(hs.size())

        output = torch.cat((output[1:, :, :],))

        return output


class ODEVAE(nn.Module):
    def __init__(self, input_size, target_size, latent_size, encoder, decoder, encoder_optimizer, decoder_optimizer,
                 loss):
        super(ODEVAE, self).__init__()
        self.input_dim = input_size
        self.output_dim = target_size
        self.hidden_dim = latent_size
        self.nodes = 28


        self.encoder = encoder
        self.decoder = decoder

        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer
        self.criterion = loss

    def forward(self, input_tensor, target_tensor):

        max_length = 500000

        input = input_tensor.reshape(input_tensor.size()[0], input_tensor.size()[1], self.nodes,
                                     int(input_tensor.size(2) / self.nodes)).squeeze(1)

        rows = input_tensor.size()[0]

        # print("tensor size", input_tensor.size())

        encoder_hidden = torch.zeros(1, rows, self.hidden_dim).cuda()

        encoder_outputs = torch.zeros(max_length, self.output_dim).cuda()
        #decoder_outputs = torch.zeros(target_tensor.size()[0], self.output_dim)

        loss = 0

        timetable = []
        for i in range(1, input_tensor.size()[0] + 1):
            for j in range(1, input_tensor.size()[1] + 1):
                timetable.append(i)

        time = torch.tensor(timetable).view(i, j).unsqueeze(-1)

        # print("input sizes to check")
        # print(input_tensor.size(), time.size())
        encoder_output, encoder_hidden = self.encoder(input, time)

        decoder_hidden = encoder_hidden

        # print("input tensor")
        # print(input_tensor.size())
        # print(time.size())
        #print(decoder_hidden.size(), "decoder hidden")

        #print("target", target_tensor.size())

        decoder_output = self.decoder(decoder_hidden, target_tensor)


        # decoder_hidden = encoder_hidden

        # print("input tensor")
        # print(input_tensor.size())
        # print(time.size())
        # print(decoder_hidden.size(), "decoder hidden")

        # decoder_output = self.decoder(decoder_hidden, target_tensor)

        return decoder_output

    def generate_with_seed(self, seed_x, t):

        input = seed_x.reshape(seed_x.size()[0], seed_x.size()[1], self.nodes,
                               int(seed_x.size(2) / self.nodes)).squeeze(1)

        seed_t_len = seed_x.shape[0]
        z_mean, z_log_var = self.encoder(input, t[:seed_t_len])
        x_p = self.decoder(z_mean, t)
        return x_p

    def getPrimitive(self, x, t):

        #print(x.size(), "x size")

        output = self.forward(x, t)
        #print("outptu size", output.size())
        prim1 = np.argmax(output[9][-1].detach().cpu())
        prim2 = np.argmax(output[19][-1].detach().cpu())
        prim3 = np.argmax(output[29][-1].detach().cpu())
        prim4 = np.argmax(output[39][-1].detach().cpu())
        prim5 = np.argmax(output[49][-1].detach().cpu())
        prim6 = np.argmax(output[59][-1].detach().cpu())
        prim7 = np.argmax(output[69][-1].detach().cpu())
        prim8 = np.argmax(output[79][-1].detach().cpu())
        prim9 = np.argmax(output[89][-1].detach().cpu())
        prim10 = np.argmax(output[99][-1].detach().cpu())
        prim11 = np.argmax(output[109][-1].detach().cpu())
        prim12 = np.argmax(output[119][-1].detach().cpu())
        prim13 = np.argmax(output[129][-1].detach().cpu())


        output = []
        output.append(prim1.tolist())
        output.append(prim2.tolist())
        output.append(prim3.tolist())
        output.append(prim4.tolist())
        output.append(prim5.tolist())
        output.append(prim6.tolist())
        output.append(prim7.tolist())
        output.append(prim8.tolist())
        output.append(prim9.tolist())
        output.append(prim10.tolist())
        output.append(prim11.tolist())
        output.append(prim12.tolist())
        output.append(prim13.tolist())


        return output
    def getCurrentPrimitive(self, x, t):

        #print(x.size(), "x size")

        output = self.forward(x, t)
        #print("outptu size", output.size())



        return np.argmax(output[-1][-1].detach().cpu())



    def train(self, input_tensor, target_tensor):


        max_length = 500

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        # input_tensor = input_tensor.unsqueeze(1)
        # target_tensor = target_tensor.unsqueeze(1)

        input = input_tensor.reshape(input_tensor.size()[0], input_tensor.size()[1], self.nodes,
                               int(input_tensor.size(2) / self.nodes)).squeeze(1)

        #input = input_tensor

        rows = input_tensor.size()[1]
        trials = input_tensor.size()[0]

        timetable = []
        for i in range(1, input_tensor.size()[0] + 1):
            for j in range(1, input_tensor.size()[1] + 1):
                timetable.append(i)

        time = torch.tensor(timetable).view(i, j).unsqueeze(-1)

        encoder_hidden = torch.zeros(trials, rows, self.hidden_dim).cuda()

        # input_length = input_tensor.size(0)
        # target_length = target_tensor.size(0)

        # print("encoder hidden")
        # print(input_tensor.size(), encoder_hidden.size())

        encoder_outputs = torch.zeros(max_length, self.output_dim).cuda()

        loss = 0
        losslist = []

        # print("input sizes to check")
        # print(input_tensor.size(), time.size())

        encoder_output, encoder_hidden = self.encoder(input, time)

        decoder_hidden = encoder_hidden

        # print("input tensor")
        # print(input_tensor.size())
        # print(time.size())
        #print(decoder_hidden.size(), "decoder hidden")

        decoder_output = self.decoder(decoder_hidden, target_tensor)

        decoder_output = decoder_output.squeeze(1)
        target_tensor = target_tensor.squeeze(1)

        labels = torch.zeros(4).cuda()

        indices = []
        for i in range(0, target_tensor.size()[0]):
            indices.append(np.argmax(target_tensor[i].cpu()))

        indicestensor = torch.tensor(indices).cuda()
        #print("indices", indicestensor.size(), decoder_output.size())

        #print(decoder_output, indicestensor)
        loss = self.criterion(decoder_output.float(), indicestensor)
        # print("encoder gru", list(self.encoder.rnn.parameters()))
        # print("decoder ode", list(self.decoder.ode.parameters()))
        # print("decoder layer", list(self.decoder.l2h.parameters()))

        loss.backward()

        #torch.nn.utils.clip_grad_norm_(self.decoder.parameters())

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item()

    def evaluate(self, input_tensor, target_tensor):

        max_length = 500

        encoder_hidden = torch.zeros(1, 1, self.hidden_dim).cuda()

        rows = input_tensor.size()[1]
        trials = input_tensor.size()[0]

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        encoder_outputs = torch.zeros(max_length, self.output_dim).cuda()

        loss = 0

        rows = input_tensor.size()[1]
        trials = input_tensor.size()[0]

        timetable = []
        for i in range(1, input_tensor.size()[0] + 1):
            for j in range(1, input_tensor.size()[1] + 1):
                timetable.append(i)

        time = torch.tensor(timetable).view(i, j).unsqueeze(-1)

        # print("input sizes to check in train")
        # print(input_tensor.size(), time.size())

        encoder_output, encoder_hidden = self.encoder(input_tensor, time)

        decoder_hidden = encoder_hidden

        # print("input tensor")
        # print(input_tensor.size())
        # print(time.size())
        # print(decoder_hidden.size(), "decoder hidden")

        decoder_output = self.decoder(decoder_hidden, target_tensor)

        # print("input tensor")
        # print(input_tensor.size())

        # print(time.size())

        # print(decoder_hidden.size(), "decoder hidden")

        decoder_output = decoder_output.squeeze(1)
        target_tensor = target_tensor.squeeze(1)

        labels = torch.zeros(4).cuda()

        indices = []
        for i in range(0, target_tensor.size()[0]):
            indices.append(np.argmax(target_tensor[i].cpu()))

        indicestensor = torch.tensor(indices).cuda()

        #print(decoder_output)

        loss = self.criterion(decoder_output.float(), indicestensor)


        return loss.item()