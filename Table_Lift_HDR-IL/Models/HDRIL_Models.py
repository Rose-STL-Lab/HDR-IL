import math

from torch.nn import functional as F

import torch
import torch.nn as nn
import utils
import numpy as np

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
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def getalpha(self, nodes):
        return F.softmax(nodes.mailbox['e'], dim=1)

    def forward(self, h):
        # equation (1)
        #print("h size", h.size())
        z = self.fc(h)
        self.g.ndata['z'] = z
        # equation (2)
        self.g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        self.g.update_all(self.message_func, self.reduce_func)

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




#With table object
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

        self.table2hid = nn.Linear(7, hidden_dim)#

        self.gru_size = self.nodes * latent_dim

        self.hid2in = nn.Linear(latent_dim, in_dim)

        self.rnn = nn.GRU(self.gru_size, self.gru_size)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.hid1 = nn.Linear(self.gru_size, hidden_dim)
        self.hid2 = nn.Linear(2*hidden_dim, hidden_dim)
        self.hid3 = nn.Linear(hidden_dim, hidden_dim)
        self.hid4 = nn.Linear(hidden_dim, hidden_dim)
        self.hid5 = nn.Linear(hidden_dim, hidden_dim)
        self.hid2hid = nn.Linear(hidden_dim, 2 * hidden_dim)



    def forward(self, h, target):

        temp = h[0]
        t = h[0, 7:14, :].reshape(1, 7)
        table = self.table2hid(t).unsqueeze(0)######

        hidden = torch.zeros((1, 1, self.gru_size)).cuda()

        for i in range(0, target.size()[0]):

            if i < h.size()[0]:
                temp = h[i]

            h1 = self.layer1(temp)

            rnn_inp = h1.flatten()

            out, hid = self.rnn(rnn_inp.unsqueeze(0).unsqueeze(0), hidden)

            out = out.reshape(self.nodes, self.latent_dim)
            hid = hid.reshape(self.nodes, self.latent_dim)
            hidden = self.layer2(hid)
            # print(hidden.size())
            hidden = hidden.flatten().unsqueeze(0).unsqueeze(0)

            temp = self.hid2in(out).squeeze(0)


        h0 = self.hid1(hidden.float())

        h0 = self.hid2(torch.cat((h0, table), 2))
        h0 = self.hid3(h0)
        h0 = self.hid2hid(h0).squeeze(0)

        zmean = h0[:, :self.hidden_dim]
        zlogvar = h0[:, self.hidden_dim:]

        return zmean, zlogvar


class GAT2(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, latent_dim, num_heads):
        super(GAT2, self).__init__()
        self.layer1 = MultiHeadGATLayer(g, in_dim, latent_dim, num_heads)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = MultiHeadGATLayer(g, latent_dim * num_heads, latent_dim, 1)

        self.input_dim = in_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.nodes = len(g.nodes())


        self.table2hid = nn.Linear(7, hidden_dim)

        self.gru_size = self.nodes * latent_dim
        self.hid2in = nn.Linear(latent_dim, in_dim)
        self.rnn = nn.GRU(self.gru_size, self.gru_size)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.hid1 = nn.Linear(self.gru_size, hidden_dim)
        self.hid2 = nn.Linear(hidden_dim, hidden_dim)
        self.hid3 = nn.Linear(hidden_dim, hidden_dim)
        self.hid4 = nn.Linear(hidden_dim, hidden_dim)
        self.hid5 = nn.Linear(hidden_dim, hidden_dim)

        self.hid2hid = nn.Linear(hidden_dim, 2 * hidden_dim)

    def forward(self, h, target):

        temp = h[0]

        hidden = torch.zeros((1, 1, self.gru_size)).cuda()

        for i in range(0, target.size()[0]):

            h1 = self.layer1(temp)

            rnn_inp = h1.flatten()
            # print(rnn_inp.size())
            out, hid = self.rnn(rnn_inp.unsqueeze(0).unsqueeze(0), hidden)

            out = out.reshape(self.nodes, self.latent_dim)
            hid = hid.reshape(self.nodes, self.latent_dim)
            hidden = self.layer2(hid)
            hidden = hidden.flatten().unsqueeze(0).unsqueeze(0)
            temp = self.hid2in(out).squeeze(0)

        h0 = self.hid1(hidden.float())

        h0 = self.hid2(h0)
        h0 = self.hid3(h0)

        h0 = self.hid2hid(h0).squeeze(0)
        zmean = h0[:, :self.hidden_dim]
        zlogvar = h0[:, self.hidden_dim:]

        return zmean, zlogvar



class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.input_dim = input_dim

        self.output_dim = int(output_dim)
        self.hidden_dim = hidden_dim


        self.gru = nn.GRU(hidden_dim, hidden_dim)

        self.bn1 = nn.BatchNorm1d(self.hidden_dim)

        self.l2h = nn.Linear(hidden_dim, hidden_dim)
        self.l2h2 = nn.Linear(hidden_dim, hidden_dim)
        self.l2h3 = nn.Linear(hidden_dim, hidden_dim)
        self.l2h4 = nn.Linear(hidden_dim, hidden_dim)
        self.l2h5 = nn.Linear(hidden_dim, hidden_dim)
        self.l2o2 = nn.Linear(hidden_dim, output_dim)

        self.l2o1 = nn.Linear(hidden_dim, self.output_dim)
        self.soft = nn.Softmax(self.output_dim)

    def forward(self, hidden, target):
        # z0 = z0.unsqueeze(1).unsqueeze(1).float().view(1,1,self.output_dim).cuda()
        hiddeninp = hidden.unsqueeze(0)

        input = torch.zeros(1, 1, hiddeninp.size()[2]).cuda()
        output = torch.zeros(1, target.size()[1], target.size()[2]).cuda()

        for i in range(0, target.size()[0]):
            zs, h = self.gru(input, hiddeninp)

            input = zs
            hs = self.l2h(zs)
            hs = self.l2h2(hs)
            hs = self.l2h3(hs)
            hs = self.l2h4(hs)
            hs = self.l2o2(hs)
            hiddeninp = h
            # softmax = self.soft(hs)

            output = torch.cat((output, hs))

        # print(hs.size())

        output = torch.cat((output[1:, :, :],))

        return output


class VAE(nn.Module):
    def __init__(self, input_size, target_size, latent_size, encoder, decoder, encoder_optimizer, decoder_optimizer,
                 loss):
        super(VAE, self).__init__()
        self.input_dim = input_size
        self.output_dim = target_size
        self.hidden_dim = latent_size


        self.encoder = encoder
        self.decoder = decoder
        self.nodes = input_size


        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer
        self.criterion = loss

    def forward(self, input_tensor, target_tensor):

        max_length = 500000

        rows = input_tensor.size()[0]

        timetable = []
        for i in range(1, target_tensor.size()[0] + 1):
            for j in range(1, target_tensor.size()[1] + 1):
                timetable.append(i)

        time = torch.tensor(timetable).view(i, j).unsqueeze(-1)
        encoder_output, encoder_hidden = self.encoder(input_tensor, time)

        decoder_hidden = encoder_hidden
        decoder_output = self.decoder(decoder_hidden, time)

        return decoder_output

    def generate_with_seed(self, seed_x, size):
        seed_t_len = seed_x.shape[0]

        print(seed_x.size(), self.nodes)
        input = seed_x.reshape(seed_x.size()[0], seed_x.size()[1], self.nodes,
                               int(seed_x.size(2) / self.nodes)).squeeze(1)


        zmean, zlogvar = self.encoder(input, size)

        z = zmean + torch.randn_like(zmean)*(zlogvar)
        x_p = self.decoder(z, size)

        return x_p


    def generate_mean_variance(self, seed_x, time, size):
        seed_t_len = seed_x.shape[0]
        #print("generate with seed", seed_x[0].size(), size[0].size(), seed_x.size(), size.size())

        dim0 = size.size()[0]
        dim1 = size.size()[1]
        dim2 = size.size()[2]

        print("dim", dim0, dim2)
        print(seed_x.size(), time.size(), "seed x")

        flat = torch.zeros([1, 1, dim0*dim2]).cuda()

        for i in range(0, 5):

            output = self.generate_with_seed(seed_x, time).cuda()
            print(output.size())

            output = output.view(1, 1, dim0*dim2)
            flat = torch.cat((flat, output), 0)


        length = flat.size()[0]
        flat = flat[1:length, :, :]


        mean = flat.mean(dim = 0)
        var = flat.var(dim = 0)
        mean = mean.view(dim0, 1, dim2)
        var = var.view(dim0, 1, dim2)

        return mean, var

    def train(self, input_tensor, target_tensor):

        max_length = 500

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        rows = target_tensor.size()[1]
        trials = target_tensor.size()[0]

        #target = target_tensor.reshape(target_tensor.size()[0], target_tensor.size()[1], nodes, int(target_tensor.size()[2]/nodes)).squeeze(1)

        timetable = []
        for i in range(1, target_tensor.size()[0] + 1):
            for j in range(1, target_tensor.size()[1] + 1):
                timetable.append(i)

        output = self.generate_with_seed(input_tensor.float(), target_tensor)

        loss = self.criterion(output.float(), target_tensor.float())

        loss.backward()

        params = list(self.parameters())


        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item()

    def evaluate(self, input_tensor, target_tensor):

        max_length = 500

        criterion = utils.DTW_Loss()

        rows = target_tensor.size()[1]
        trials = target_tensor.size()[0]

        output = self.generate_with_seed(input_tensor.float(), target_tensor)

        print(output.size(), target_tensor.size())
        loss = criterion(output.float(), target_tensor.float())



        return loss.item()






class PlanningGAT(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, latent_dim, num_heads):
        super(PlanningGAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(g, in_dim, latent_dim, num_heads)

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

        # print("h0", h0.size())

        #h0 = self.hid2hid(h0)


        #zmean = h0

        h0 = h0
        print("h0", h0.size(), self.hidden_dim)



        return out, h0



#Planning Model
class PlanningDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(PlanningDecoder, self).__init__()
        self.input_dim = input_dim

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim


        self.gru = nn.GRU(hidden_dim, hidden_dim)

        self.bn1 = nn.BatchNorm1d(self.hidden_dim)

        self.l2h = nn.Linear(hidden_dim, hidden_dim)
        self.l2h2 = nn.Linear(hidden_dim, hidden_dim)
        self.l2h3 = nn.Linear(hidden_dim, hidden_dim)
        self.l2h4 = nn.Linear(hidden_dim, output_dim)
        self.soft = nn.Softmax(2)

    def forward(self, hidden, target):

        print(target.size())
        input = torch.zeros(1, 1, hidden.size()[2]).cuda()
        output = torch.zeros(1, target.size()[1], target.size()[2]).cuda()

        for i in range(0, target.size()[0]):
            zs, h = self.gru(input, hidden)

            input = zs

            zs = self.l2h(zs)
            zs = self.l2h2(zs)
            zs = self.l2h3(zs)
            hs = self.l2h4(zs)

            hidden = h

            softmax = self.soft(hs)

            output = torch.cat((output, softmax))

        output = torch.cat((output[1:, :, :],))

        return output

class PlanningVAE(nn.Module):
    def __init__(self, input_size, target_size, latent_size, encoder, decoder, encoder_optimizer, decoder_optimizer,
                 loss):

        super(PlanningVAE, self).__init__()
        self.input_dim = input_size
        self.output_dim = target_size
        self.hidden_dim = latent_size
        self.nodes = input_size

        self.encoder = encoder
        self.decoder = decoder

        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer
        self.criterion = loss

    def forward(self, input_tensor, target_tensor):

        input = input_tensor.reshape(input_tensor.size()[0], input_tensor.size()[1], self.nodes,
                                     int(input_tensor.size(2) / self.nodes)).squeeze(1)

        timetable = []
        for i in range(1, input_tensor.size()[0] + 1):
            for j in range(1, input_tensor.size()[1] + 1):
                timetable.append(i)

        time = torch.tensor(timetable).view(i, j).unsqueeze(-1)

        encoder_output, encoder_hidden = self.encoder(input, time)

        decoder_hidden = encoder_hidden


        decoder_output = self.decoder(decoder_hidden, target_tensor)



        return decoder_output

    def generate_with_seed(self, seed_x, t):

        input = seed_x.reshape(seed_x.size()[0], seed_x.size()[1], self.nodes,
                               int(seed_x.size(2) / self.nodes)).squeeze(1)

        seed_t_len = seed_x.shape[0]
        z_mean, z_log_var = self.encoder(input, t[:seed_t_len])
        x_p = self.decoder(z_mean, t)
        return x_p

    def getPrimitive(self, x, t):

        # print(x.size(), "x size")

        output = self.forward(x, t)
        # print("outptu size", output.size())
        prim1 = np.argmax(output[9][-1].detach().cpu())
        prim2 = np.argmax(output[21][-1].detach().cpu())
        prim3 = np.argmax(output[33][-1].detach().cpu())
        prim4 = np.argmax(output[45][-1].detach().cpu())
        prim5 = np.argmax(output[57][-1].detach().cpu())
        prim6 = np.argmax(output[69][-1].detach().cpu())

        output = []
        output.append(prim1.tolist())
        output.append(prim2.tolist())
        output.append(prim3.tolist())
        output.append(prim4.tolist())
        output.append(prim5.tolist())
        output.append(prim6.tolist())

        return output

    def getCurrentPrimitive(self, x, t):


        output = self.forward(x, t)

        return np.argmax(output[-1][-1].detach().cpu())

    def train(self, input_tensor, target_tensor):

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()


        input = input_tensor.reshape(input_tensor.size()[0], input_tensor.size()[1], self.nodes,
                                     int(input_tensor.size(2) / self.nodes)).squeeze(1)



        timetable = []
        for i in range(1, input_tensor.size()[0] + 1):
            for j in range(1, input_tensor.size()[1] + 1):
                timetable.append(i)

        time = torch.tensor(timetable).view(i, j).unsqueeze(-1)

        encoder_output, encoder_hidden = self.encoder(input, time)

        decoder_hidden = encoder_hidden

        decoder_output = self.decoder(decoder_hidden, target_tensor)

        decoder_output = decoder_output.squeeze(1)
        target_tensor = target_tensor.squeeze(1)

        indices = []
        for i in range(0, target_tensor.size()[0]):
            indices.append(np.argmax(target_tensor[i].cpu()))

        indicestensor = torch.tensor(indices).cuda()

        loss = self.criterion(decoder_output.float(), indicestensor)

        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item()

