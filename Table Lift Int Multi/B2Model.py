import math

import inline as inline
import matplotlib
import numpy as np
from IPython.display import clear_output
#from tqdm import tqdm_notebook as tqdm

from dgl.nn.pytorch import GATConv

import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns

sns.color_palette("bright")
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torch import Tensor, optim
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

import math
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn

use_cuda = torch.cuda.is_available()


def ode_solve(z0, t0, t1, f):
    """
    Simplest Euler ODE initial value solver
    """

    h_max = 5
    n_steps = math.ceil((abs(t1 - t0) / h_max).max().item())

    h = ((t1 - t0) / n_steps)
    t = t0
    z = z0

    """
        print("ode solver parameters")
        print(z0.size())
        print(t0.size())
        print("end ode parameters")
    """

    for i_step in range(n_steps):
        """
        print(i_step)
        print("z")
        print(z.size())
        print("t")
        print(t.size())
        print("h")
        print(h.size())
        print("f(z,t)")
        print(f(z,t).size())
        print("h*f(z,t)")
        print((h * f(z,t)).size())
        """

        z = z + h * f(z, t)
        t = t + h

    return z


"""tensor 1 has multiple dimensions, tensor 2 is of rank 1"""


def tensorMultiply(tensor1, tensor2):
    s = []
    trans = tensor1.t()

    for row in trans:
        r = row.unsqueeze(1)
        a = tensor2 * r
        s.append(a.tolist())

    values = np.asarray(s)

    output = np.sum(values, axis=0)

    return torch.tensor(output)


class ODEF(nn.Module):
    def forward_with_grad(self, z, t, grad_outputs):
        """Compute f and a df/dz, a df/dp, a df/dt"""
        batch_size = z.shape[0]

        out = self.forward(z, t)

        a = grad_outputs
        adfdz, adfdt, *adfdp = torch.autograd.grad(
            (out,), (z, t) + tuple(self.parameters()), grad_outputs=(a),
            allow_unused=True, retain_graph=True
        )
        # grad method automatically sums gradients for batch items, we have to expand them back
        if adfdp is not None:
            adfdp = torch.cat([p_grad.flatten() for p_grad in adfdp]).unsqueeze(0)
            adfdp = adfdp.expand(batch_size, -1) / batch_size
        if adfdt is not None:
            adfdt = adfdt.expand(batch_size, 1) / batch_size
        return out, adfdz, adfdt, adfdp

    def flatten_parameters(self):
        p_shapes = []
        flat_parameters = []
        for p in self.parameters():
            p_shapes.append(p.size())
            flat_parameters.append(p.flatten())
        return torch.cat(flat_parameters)


class ODEAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z0, t, flat_parameters, func):

        assert isinstance(func, ODEF)
        bs, *z_shape = z0.size()
        time_len = t.size(0)

        with torch.no_grad():
            z = torch.zeros(time_len, bs, *z_shape).to(z0)
            z[0] = z0

            for i_t in range(time_len - 1):
                z0 = ode_solve(z0, t[i_t], t[i_t + 1], func)
                z[i_t + 1] = z0

        ctx.func = func
        ctx.save_for_backward(t, z.clone(), flat_parameters)

        return z

    @staticmethod
    def backward(ctx, dLdz):
        """
        dLdz shape: time_len, batch_size, *z_shape
        """

        func = ctx.func
        t, z, flat_parameters = ctx.saved_tensors

        time_len, bs, *z_shape = z.size()
        n_dim = np.prod(z_shape)
        n_params = flat_parameters.size(0)

        # Dynamics of augmented system to be calculated backwards in time
        def augmented_dynamics(aug_z_i, t_i):
            """
            tensors here are temporal slices
            t_i - is tensor with size: bs, 1
            aug_z_i - is tensor with size: bs, n_dim*2 + n_params + 1
            """

            """
            print("aug_z_i, t_i")
            print(aug_z_i.size())
            print(t_i.size())
            """

            z_i, a = aug_z_i[:, :n_dim], aug_z_i[:, n_dim:2 * n_dim]  # ignore parameters and time

            # Unflatten z and a
            z_i = z_i.view(bs, *z_shape)
            a = a.view(bs, *z_shape)
            with torch.set_grad_enabled(True):
                t_i = t_i.detach().requires_grad_(True)
                z_i = z_i.detach().requires_grad_(True)
                func_eval, adfdz, adfdt, adfdp = func.forward_with_grad(z_i, t_i, grad_outputs=a)  # bs, *z_shape
                adfdz = adfdz.to(z_i) if adfdz is not None else torch.zeros(bs, *z_shape).to(z_i)
                adfdp = adfdp.to(z_i) if adfdp is not None else torch.zeros(bs, n_params).to(z_i)
                adfdt = adfdt.to(z_i) if adfdt is not None else torch.zeros(bs, 1).to(z_i)

            # Flatten f and adfdz
            func_eval = func_eval.view(bs, n_dim)
            adfdz = adfdz.view(bs, n_dim)

            return torch.cat((func_eval, -adfdz, -adfdp, -adfdt), dim=1)

        dLdz = dLdz.view(time_len, bs, n_dim)  # flatten dLdz for convenience
        with torch.no_grad():
            ## Create placeholders for output gradients
            # Prev computed backwards adjoints to be adjusted by direct gradients
            adj_z = torch.zeros(bs, n_dim).to(dLdz)
            adj_p = torch.zeros(bs, n_params).to(dLdz)
            # In contrast to z and p we need to return gradients for all times
            adj_t = torch.zeros(time_len, bs, 1).to(dLdz)

            for i_t in range(time_len - 1, 0, -1):
                z_i = z[i_t]
                t_i = t[i_t]
                f_i = func(z_i, t_i).view(bs, n_dim)

                # Compute direct gradients
                dLdz_i = dLdz[i_t]
                dLdt_i = torch.bmm(torch.transpose(dLdz_i.unsqueeze(-1), 1, 2), f_i.unsqueeze(-1))[:, 0]

                # Adjusting adjoints with direct gradients
                adj_z += dLdz_i
                adj_t[i_t] = adj_t[i_t] - dLdt_i

                # Pack augmented variable
                aug_z = torch.cat((z_i.view(bs, n_dim), adj_z, torch.zeros(bs, n_params).to(z), adj_t[i_t]), dim=-1)

                # Solve augmented system backwards
                aug_ans = ode_solve(aug_z, t_i, t[i_t - 1], augmented_dynamics)

                # Unpack solved backwards augmented system
                adj_z[:] = aug_ans[:, n_dim:2 * n_dim]
                adj_p[:] += aug_ans[:, 2 * n_dim:2 * n_dim + n_params]
                adj_t[i_t - 1] = aug_ans[:, 2 * n_dim + n_params:]

                del aug_z, aug_ans

            ## Adjust 0 time adjoint with direct gradients
            # Compute direct gradients
            dLdz_0 = dLdz[0]
            dLdt_0 = torch.bmm(torch.transpose(dLdz_0.unsqueeze(-1), 1, 2), f_i.unsqueeze(-1))[:, 0]

            # Adjust adjoints
            adj_z += dLdz_0
            adj_t[0] = adj_t[0] - dLdt_0

        return adj_z.view(bs, *z_shape), adj_t, adj_p, None


class NeuralODE(nn.Module):
    def __init__(self, func):
        super(NeuralODE, self).__init__()
        assert isinstance(func, ODEF)
        self.func = func

    def forward(self, z0, t=Tensor([0., 1.]), return_whole_sequence=False):
        t = t.to(z0)

        z = ODEAdjoint.apply(z0, t, self.func.flatten_parameters(), self.func)

        # print("z values")
        # print(t.size(), z0.size())
        # print(z.size(), z[-1].size())

        if return_whole_sequence:
            return z
        else:
            return z[-1]


class LinearODEF(ODEF):
    def __init__(self, W):
        super(LinearODEF, self).__init__()
        self.lin = nn.Linear(2, 2, bias=False)
        self.lin.weight = nn.Parameter(W)

    def forward(self, x, t):
        return self.lin(x)


class SpiralFunctionExample(LinearODEF):
    def __init__(self):
        super(SpiralFunctionExample, self).__init__(Tensor([[-0.1, -1.], [1., -0.1]]))


class RandomLinearODEF(LinearODEF):
    def __init__(self):
        super(RandomLinearODEF, self).__init__(torch.randn(2, 2) / 2.)


class NNODEF(ODEF):
    def __init__(self, in_dim, hid_dim, time_invariant=False):
        super(NNODEF, self).__init__()

        self.time_invariant = time_invariant

        if time_invariant:
            self.lin1 = nn.Linear(in_dim, hid_dim)
        else:
            self.lin1 = nn.Linear(in_dim + 1, hid_dim)

        self.lin2 = nn.Linear(hid_dim, hid_dim)
        self.lin3 = nn.Linear(hid_dim, in_dim)
        self.elu = nn.ELU(inplace=True)

    def forward(self, x, t):

        x = x.float()
        t = t.float()
        if not self.time_invariant:
            x = torch.cat((x, t), dim=-1)

        h = self.elu(self.lin1(x))
        h = self.elu(self.lin2(h))
        out = self.lin3(h)
        return out


def to_np(x):
    return x.detach().cpu().numpy()


def plot_trajectories(obs=None, times=None, trajs=None, save=None, figsize=(16, 8)):
    plt.figure(figsize=figsize)
    if obs is not None:
        if times is None:
            times = [None] * len(obs)
        for o, t in zip(obs, times):
            o, t = NNODEF.to_np(o), NNODEF.to_np(t)
            for b_i in range(o.shape[1]):
                plt.scatter(o[:, b_i, 0], o[:, b_i, 1], c=t[:, b_i, 0], cmap=cm.plasma)

    if trajs is not None:
        for z in trajs:
            z = NNODEF.to_np(z)
            plt.plot(z[:, 0, 0], z[:, 0, 1], lw=1.5)
        if save is not None:
            plt.savefig(save)
    plt.show()


def conduct_experiment(ode_true, ode_trained, n_steps, name, plot_freq=10):
    # Create data
    z0 = Variable(torch.Tensor([[0.6, 0.3]]))

    t_max = 6.29 * 5
    n_points = 200

    index_np = np.arange(0, n_points, 1, dtype=np.int)
    index_np = np.hstack([index_np[:, None]])
    times_np = np.linspace(0, t_max, num=n_points)
    times_np = np.hstack([times_np[:, None]])

    times = torch.from_numpy(times_np[:, :, None]).to(z0)
    obs = ode_true(z0, times, return_whole_sequence=True).detach()
    obs = obs + torch.randn_like(obs) * 0.01

    # Get trajectory of random timespan
    min_delta_time = 1.0
    max_delta_time = 5.0
    max_points_num = 32

    def create_batch():

        t0 = np.random.uniform(0, t_max - max_delta_time)
        t1 = t0 + np.random.uniform(min_delta_time, max_delta_time)

        idx = sorted(np.random.permutation(index_np[(times_np > t0) & (times_np < t1)])[:max_points_num])

        obs_ = obs[idx]
        ts_ = times[idx]
        return obs_, ts_

    # Train Neural ODE
    optimizer = torch.optim.Adam(ode_trained.parameters(), lr=0.01)
    for i in range(n_steps):
        obs_, ts_ = create_batch()

        z_ = ode_trained(obs_[0], ts_, return_whole_sequence=True)
        loss = F.mse_loss(z_, obs_.detach())

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        if i % plot_freq == 0:
            z_p = ode_trained(z0, times, return_whole_sequence=True)

            NNODEF.plot_trajectories(obs=[obs], times=[times], trajs=[z_p], save=f"assets/imgs/{name}/{i}.png")
            clear_output(wait=True)


def norm(dim):
    return nn.BatchNord2d(dim)


def conv3x3(in_feats, out_feats, stride=1):
    return nn.Conv2d(in_feats, out_feats, kernel_size=3, stride=stride, padding=1, bias=False)


def add_time(in_tensor, t):
    bs, c, w, h = in_tensor.shape
    return torch.cat((in_tensor, t.expand(bs, 1, w, h)), dim=1)


class FuncODEF(ODEF):
    def __init__(self, input_size, hidden_size, output_size):
        super(FuncODEF, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x, t):
        xt = add_time(x, t)
        h = self.norm1(torch.relu(self.fc1(xt)))
        ht = add_time(h, t)
        dxdt = self.norm2(torch.relu(self.conv2(ht)))
        return dxdt


def norm(dim):
    return nn.BatchNorm2d(dim)


def conv3x3(in_feats, out_feats, stride=1):
    return nn.Conv2d(in_feats, out_feats, kernel_size=3, stride=stride, padding=1, bias=False)


def add_time(in_tensor, t):
    bs, c, w, h = in_tensor.shape
    return torch.cat((in_tensor, t.expand(bs, 1, w, h)), dim=1)


class NeuralODEF(ODEF):
    def __init__(self, dim):
        super(NeuralODEF, self).__init__()
        self.conv1 = conv3x3(dim + 1, dim)
        self.norm1 = norm(dim)
        self.conv2 = conv3x3(dim + 1, dim)
        self.norm2 = norm(dim)

    def forward(self, x, t):
        xt = add_time(x, t)
        h = self.norm1(torch.relu(self.conv1(xt)))
        ht = add_time(h, t)
        dxdt = self.norm2(torch.relu(self.conv2(ht)))
        return dxdt





def reset_param(t):
    stdv = 2. / math.sqrt(t.size(0))
    t.data.uniform_(-stdv, stdv)


class GCNLayer(torch.nn.Module):
    # GCN layer for a single time point
    # Implemented for sparse matrices
    def __init__(self, node_repr_dim, output_dim, activation):
        super().__init__()
        self.activation = activation
        self.input_node_dim = node_repr_dim
        self.output_dim = output_dim

        self.w = Parameter(torch.Tensor(node_repr_dim, output_dim)).float().cuda()
        reset_param(self.w)  # initialize as specified in reset_param

    def forward(self, Ahat, node_feats):

        # Ahat: [NxN] adjacency matrix with self-loops
        # node_feats: [N x self.input_node_dim] matrix
        nodefeats = node_feats.matmul(self.w).float()
        ah = Ahat.float().matmul(nodefeats)
        last_l = self.activation(ah)
        return last_l


class StaticGCN(torch.nn.Module):
    # Static GCN model to embed graph
    def __init__(self, args, activation):
        super().__init__()

        # args: dict with the following keys:
        #    'num_layers' (int) number of GCN layers
        #    'feats_per_node' (int) number of features in input data
        #    'layer_node_feats' (list of ints) for each layer, the input features size
        # for example, for 2 layers of GCN with 8 input features, and we want to embed
        # each node into 10 dimensions in layer 0, then to 15 in layer 1, and have an output of dimension 8,
        # args = {'num_layers': 3, 'feats_per_node': 8, 'layer_node_feats': [10, 15, 8]}

        self.num_gcn_layers = args['num_layers']
        self.gcn = nn.ModuleList()  # list of GCN layers

        # GCN layers
        for l in range(self.num_gcn_layers):
            if l == 0:
                node_repr_dim = args['feats_per_node']
            else:
                node_repr_dim = args['layer_node_feats'][l - 1]
            output_dim = args['layer_node_feats'][l]
            self.gcn.append(GCNLayer(node_repr_dim, output_dim, activation))

        self.nodedim = node_repr_dim

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, Ahat_list, node_feats_list):
        # Ahat_list is a list of either one tensor (for regular GCN) or
        #  a list of a sequence of tensors for a sequence of graphs
        max_t = len(Ahat_list)
        emb_list = []
        for t in range(max_t):
            node_repr = node_feats_list[t]
            for l in range(self.num_gcn_layers):
                h_t = self.gcn[l](Ahat_list[t], node_repr)
                node_repr = h_t
            emb_list.append(h_t)
        return emb_list


class TemporalGCN(StaticGCN):
    # GCN with temporal model for embeddings.
    # Cannot handle changing node set
    def __init__(self, args, activation, input_dim, hidden_dim):
        super().__init__(args, activation)

        self.hidden_dim = hidden_dim



        self.rnn = nn.GRU(input_size = self.nodedim, hidden_size = hidden_dim, num_layers = self.num_gcn_layers)
        self.hid1 = nn.Linear(hidden_dim, hidden_dim)
        self.hid2 = nn.Linear(hidden_dim, hidden_dim)
        self.hid3 = nn.Linear(hidden_dim, hidden_dim)
        self.hid4 = nn.Linear(hidden_dim, hidden_dim)
        self.hid2inp = nn.Linear(hidden_dim, self.nodedim)

        self.hid2hid = nn.Linear(hidden_dim, 2 * hidden_dim)

        # Temporal model for embeddings
        #self.rnn = nn.LSTM(input_size=args.layer_2_feats, hidden_size=args.lstm_l2_feats, num_layers=args.lstm_l2_layers)

    def forward(self, Ahat_list, node_feats_list):
        # graphs at every time point must be the same size
        # Ahat_list is a list of adjacency matrices in a sequence
        # node_feats_list is a list of node features in a sequence


        num_nodes = Ahat_list[0].shape[0]

        max_t = len(Ahat_list)
        h_t_seq = []
        emb_list = []



        for t in range(max_t):
            #print(t)

            if (t < node_feats_list.size()[0]):
                h_t = node_feats_list[t]  # initial embedding at t
            else:
                h_t = self.hid2inp(h_t)

            assert Ahat_list[t].size() == (num_nodes, num_nodes) and \
                   h_t[0].shape[0] == num_nodes, 'Graph size cannot change across time for Temporal GCN '


            for gcn_layer in self.gcn:
                h_t = gcn_layer(Ahat_list[t], h_t.float())
            h_t_seq.append(h_t)

            graph_embeddings = torch.stack(h_t_seq)

            out_embedding, h_t = self.rnn(graph_embeddings.squeeze(1))  # should new graph be input?

        #print(h_t.size(), "ht size")
        h0 = self.hid1(h_t[0].float())
        h0 = self.hid2(h0)
        h0 = self.hid3(h0)
        h0 = self.hid4(h0)
        #h0 = self.hid5(h0)

        h0 = self.hid2hid(h0)
        zmean = h0[:, :self.hidden_dim]
        zlogvar = h0[:, self.hidden_dim:]

        return zmean, zlogvar










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
        #print(alpha.reshape(21, 1, 21))
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
        #self.layer1 = MultiHeadGATLayer(g, in_dim, latent_dim, num_heads)
        self.layer1 = nn.Linear(in_dim, latent_dim)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = nn.Linear(latent_dim * num_heads, latent_dim)

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
        self.hid2 = nn.Linear(2*hidden_dim, hidden_dim)
        self.hid3 = nn.Linear(hidden_dim, hidden_dim)
        self.hid4 = nn.Linear(hidden_dim, hidden_dim)
        self.hid5 = nn.Linear(hidden_dim, hidden_dim)

        self.hid2hid = nn.Linear(hidden_dim, 2 * hidden_dim)

    def forward(self, h, target):
        #print(h[0, 14:21, :].size())

        temp = h[0]
        t = h[0, 14:21, :].reshape(1, 7)
        table = self.table2hid(t).unsqueeze(0)
        #print("table")

        hidden = torch.zeros((1, 1, self.gru_size)).cuda()



        for i in range(0, target.size()[0]):

            if i < h.size()[0]:
                temp = h[i]


            h1 = self.layer1(temp)
            #h1 = F.elu(h1)
            #h2 = self.layer2(h1)
            #print("rnn input", h2.size())

            rnn_inp = h1.flatten()
            # print(rnn_inp.size())

            out, hid = self.rnn(rnn_inp.unsqueeze(0).unsqueeze(0), hidden)

            out = out.reshape(self.nodes, self.latent_dim)
            hid = hid.reshape(self.nodes, self.latent_dim)
            hidden = self.layer2(hid)
            # print(hidden.size())
            hidden = hidden.flatten().unsqueeze(0).unsqueeze(0)
            #hidden = hid

            temp = self.hid2in(out).squeeze(0)
            #print(out.size(), "output size")

        h0 = self.hid1(hidden.float())
        #print(table.size(), h0.size(), torch.cat((h0, table), 2).size())
        h0 = self.hid2(torch.cat((h0, table), 2))
        h0 = self.hid3(h0)

        #print("h0", h0.size())

        h0 = self.hid2hid(h0).squeeze(0)

        #print("h0", h0.size(), self.hidden_dim)
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

        #print("in dim", in_dim)


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
        #print(h[0, 14:21, :].size())

        temp = h[0]
        #t = h[0, 14:21, :].reshape(1, 7)
        #table = self.table2hid(t).unsqueeze(0)
        #print("table")

        #print(target.size(), )

        hidden = torch.zeros((1, 1, self.gru_size)).cuda()



        for i in range(0, target.size()[0]):


            if i < h.size()[0]:
                temp = h[i]
            #print("temp", temp.size())

            h1 = self.layer1(temp)
            #h2 = F.elu(h1)

            #h2 = self.layer2(h2)

            #print("rnn input", h2.size())
            #print("h1", h1.size())

            rnn_inp = h1.flatten()
            # print(rnn_inp.size())

            out, hid = self.rnn(rnn_inp.unsqueeze(0).unsqueeze(0), hidden)
            #print("out size", out.size(), hid.size(), rnn_inp.size())


            out = out.reshape(self.nodes, self.latent_dim)
            hid = hid.reshape(self.nodes, self.latent_dim)
            hidden = self.layer2(hid)
            #print(hidden.size())
            hidden = hidden.flatten().unsqueeze(0).unsqueeze(0)
            #hidden = hid
            temp = self.hid2in(out).squeeze(0)
            #print(out.size(), "output size")




        h0 = self.hid1(hidden.float())
        #print(table.size(), h0.size(), torch.cat((h0, table), 2).size())
        h0 = self.hid2(h0)
        h0 = self.hid3(h0)

        #print("h0", h0.size())


        h0 = self.hid2hid(h0).squeeze(0)

        #print("h0", h0.size(), self.hidden_dim)
        zmean = h0[:, :self.hidden_dim]
        zlogvar = h0[:, self.hidden_dim:]

        return zmean, zlogvar







class RNNEncoder(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim):
        super(RNNEncoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.rnn = nn.GRU(input_dim + 1, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.hid1 = nn.Linear(hidden_dim, hidden_dim)
        self.hid2 = nn.Linear(hidden_dim, hidden_dim)
        self.hid3 = nn.Linear(hidden_dim, hidden_dim)
        self.hid4 = nn.Linear(hidden_dim, hidden_dim)
        self.hid5 = nn.Linear(hidden_dim, hidden_dim)

        self.hid2hid = nn.Linear(hidden_dim, 2 * hidden_dim)


    def forward(self, x, time):

        #x= torch.cat((x.float(), time.float().cuda()), dim=-1)

        x = x.float().cuda()

        output, h0 = self.rnn(x)  # Reversed
        h0 = self.hid1(h0[0].float())
        h0 = self.hid2(h0)
        h0 = self.hid3(h0)
        h0 = self.hid4(h0)
        h0 = self.hid5(h0)


        h0 = self.hid2hid(h0)

        zmean = h0[:, :self.hidden_dim]
        zlogvar = h0[:, self.hidden_dim:]



        return zmean, zlogvar


class NeuralODEDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(NeuralODEDecoder, self).__init__()
        self.input_dim = input_dim

        self.output_dim = int(output_dim)
        self.hidden_dim = hidden_dim

        func = NNODEF(hidden_dim, hidden_dim, time_invariant=True)
        self.ode = NeuralODE(func)
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

        # print("decoder forward hidden", hidden.size())

        # z0 = z0.unsqueeze(1).unsqueeze(1).float().view(1,1,self.output_dim).cuda()
        # z0 = z0.unsqueeze(1).unsqueeze(1).float().view(1,1,self.output_dim).cuda()
        hiddeninp = hidden.unsqueeze(0)

        input = torch.zeros(1, 1, hiddeninp.size()[2]).cuda()
        output = torch.zeros(1, target.size()[1], target.size()[2]).cuda()

        for i in range(0, target.size()[0]):
            zs, h = self.gru(input, hiddeninp)
            # print("decoder forward hidden", hidden.size(), time.size(), zs.size())
            # print(hidden)
            # print(time)
            # print(zs)

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


class ODEVAE(nn.Module):
    def __init__(self, input_size, target_size, latent_size, encoder, decoder, encoder_optimizer, decoder_optimizer,
                 loss):
        super(ODEVAE, self).__init__()
        self.input_dim = input_size
        self.output_dim = target_size
        self.hidden_dim = latent_size


        self.encoder = encoder
        self.decoder = decoder
        self.nodes = 21


        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer
        self.criterion = loss

    def forward(self, input_tensor, target_tensor):

        max_length = 500000

        rows = input_tensor.size()[0]

        # print("tensor size", input_tensor.size())

        encoder_hidden = torch.zeros(1, rows, self.hidden_dim).cuda()

        encoder_outputs = torch.zeros(max_length, self.output_dim).cuda()
        decoder_outputs = torch.zeros(target_tensor.size()[0], self.output_dim)

        loss = 0

        timetable = []
        for i in range(1, target_tensor.size()[0] + 1):
            for j in range(1, target_tensor.size()[1] + 1):
                timetable.append(i)

        time = torch.tensor(timetable).view(i, j).unsqueeze(-1)

        # print("input sizes to check")
        # print(input_tensor.size(), time.size())
        encoder_output, encoder_hidden = self.encoder(input_tensor, time)

        decoder_hidden = encoder_hidden

        # print("input tensor")
        # print(input_tensor.size())
        # print(time.size())
        # print(decoder_hidden.size(), "decoder hidden")

        decoder_output = self.decoder(decoder_hidden, time)

        return decoder_output

    def generate_with_seed(self, seed_x, size):
        seed_t_len = seed_x.shape[0]
        #print("generate with seed", seed_x[0].size(), size[0].size(), seed_x.size(), size.size())

        input = seed_x.reshape(seed_x.size()[0], seed_x.size()[1], self.nodes,
                               int(seed_x.size(2) / self.nodes)).squeeze(1)


        #seed_x_reshape = seed_x.reshape(seed_x.size(0), seed_x.size(1), nodes, int(seed_x.size()[2]/nodes))


        #print("forawrd", seed_x_reshape.size())
        zmean, zlogvar = self.encoder(input, size)

        #print("zmean", zmean.size(), zlogvar.size())

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

        for i in range(0, 30):

            output = self.generate_with_seed(seed_x, time).cuda()
            print(output.size())

            output = output.view(1, 1, dim0*dim2)
            flat = torch.cat((flat, output), 0)


        length = flat.size()[0]
        flat = flat[1:length, :, :]

        #print(flat.size(), output.size())

        mean = flat.mean(dim = 0)
        var = flat.var(dim = 0)
        mean = mean.view(dim0, 1, dim2)
        var = var.view(dim0, 1, dim2)

        return mean, var

    def train(self, input_tensor, target_tensor):

        max_length = 500

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        # input_tensor = input_tensor.unsqueeze(1)
        # target_tensor = target_tensor.unsqueeze(1)

        rows = target_tensor.size()[1]
        trials = target_tensor.size()[0]


        #target = target_tensor.reshape(target_tensor.size()[0], target_tensor.size()[1], nodes, int(target_tensor.size()[2]/nodes)).squeeze(1)

        timetable = []
        for i in range(1, target_tensor.size()[0] + 1):
            for j in range(1, target_tensor.size()[1] + 1):
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

        # encoder_output, encoder_hidden = self.encoder(input_tensor.float(), time.float())

        # decoder_hidden = encoder_hidden

        # print("input tensor")
        # print(input_tensor.size())
        # print(time.size())
        # print(decoder_hidden.size(), "decoder hidden")

        # decoder_output = self.decoder(decoder_hidden, time)

        output = self.generate_with_seed(input_tensor.float(), target_tensor)

        #print(output.size(), target_tensor.size())
        loss = self.criterion(output.float(), target_tensor.float())
        # print("encoder gru", list(self.encoder.rnn.parameters()))
        # print("decoder ode", list(self.decoder.ode.parameters()))
        # print("decoder layer", list(self.decoder.l2h.parameters()))

        loss.backward()


        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item()

    def evaluate(self, input_tensor, target_tensor):

        max_length = 500
        import differentiableDP as dtw
        criterion = dtw.DTW_Loss()

        # input_tensor = input_tensor.unsqueeze(1)
        # target_tensor = target_tensor.unsqueeze(1)

        rows = target_tensor.size()[1]
        trials = target_tensor.size()[0]

        # target = target_tensor.reshape(target_tensor.size()[0], target_tensor.size()[1], nodes, int(target_tensor.size()[2]/nodes)).squeeze(1)


        # print("input tensor")
        # print(input_tensor.size())
        # print(time.size())
        # print(decoder_hidden.size(), "decoder hidden")

        # decoder_output = self.decoder(decoder_hidden, time)

        output = self.generate_with_seed(input_tensor.float(), target_tensor)

        print(output.size(), target_tensor.size())
        loss = criterion(output.float(), target_tensor.float())
        # print("encoder gru", list(self.encoder.rnn.parameters()))
        # print("decoder ode", list(self.decoder.ode.parameters()))
        # print("decoder layer", list(self.decoder.l2h.parameters()))


        return loss.item()