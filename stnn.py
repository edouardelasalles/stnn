import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from module import MLP
from utils import identity


class SaptioTemporalNN(nn.Module):
    def __init__(self, relations, nx, nt, nd, nz, mode=None, nhid=0, nlayers=1, dropout_f=0., dropout_d=0.,
                 activation='tanh', periode=1):
        super(SaptioTemporalNN, self).__init__()
        assert (nhid > 0 and nlayers > 1) or (nhid == 0 and nlayers == 1)
        # attributes
        self.nt = nt
        self.nx = nx
        self.nz = nz
        self.mode = mode
        # kernel
        self.activation = F.tanh if activation == 'tanh' else identity if activation == 'identity' else None
        device = relations.device
        if mode is None or mode == 'refine':
            self.relations = torch.cat((torch.eye(nx).to(device).unsqueeze(1), relations), 1)
        elif mode == 'discover':
            self.relations = torch.cat((torch.eye(nx).to(device).unsqueeze(1),
                                        torch.ones(nx, 1, nx).to(device)), 1)
        self.nr = self.relations.size(1)
        # modules
        self.drop = nn.Dropout(dropout_f)
        self.factors = nn.Parameter(torch.Tensor(nt, nx, nz))
        self.dynamic = MLP(nz * self.nr, nhid, nz, nlayers, dropout_d)
        self.decoder = nn.Linear(nz, nd, bias=False)
        if mode == 'refine':
            self.relations.data = self.relations.data.ceil().clamp(0, 1).byte()
            self.rel_weights = nn.Parameter(torch.Tensor(self.relations.sum().item() - self.nx))
        elif mode == 'discover':
            self.rel_weights = nn.Parameter(torch.Tensor(nx, 1, nx))
        # init
        self._init_weights(periode)

    def _init_weights(self, periode):
        initrange = 0.1
        if periode >= self.nt:
            self.factors.data.uniform_(-initrange, initrange)
        else:
            timesteps = torch.arange(self.factors.size(0)).long()
            for t in range(periode):
                idx = timesteps % periode == t
                idx_data = idx.view(-1, 1, 1).expand_as(self.factors)
                init = torch.Tensor(self.nx, self.nz).uniform_(-initrange, initrange).repeat(idx.sum().item(), 1, 1)
            self.factors.data.masked_scatter_(idx_data, init.view(-1))
        if self.mode == 'refine':
            self.rel_weights.data.fill_(0.5)
        elif self.mode == 'discover':
            self.rel_weights.data.fill_(1 / self.nx)

    def get_relations(self):
        if self.mode is None:
            return self.relations
        else:
            weights = F.hardtanh(self.rel_weights, 0, 1)
            if self.mode == 'refine':
                intra = self.rel_weights.new(self.nx, self.nx).copy_(self.relations[:, 0]).unsqueeze(1)
                inter = self.rel_weights.new_zeros(self.nx, self.nr - 1, self.nx)
                inter.masked_scatter_(self.relations[:, 1:], weights)
            if self.mode == 'discover':
                intra = self.relations[:, 0].unsqueeze(1)
                inter = weights
            return torch.cat((intra, inter), 1)

    def update_z(self, z):
        z_context = self.get_relations().matmul(z).view(-1, self.nr * self.nz)
        z_next = self.dynamic(z_context)
        return self.activation(z_next)

    def decode_z(self, z):
        x_rec = self.decoder(z)
        return x_rec

    def dec_closure(self, t_idx, x_idx):
        z_inf = self.drop(self.factors[t_idx, x_idx])
        x_rec = self.decoder(z_inf)
        return x_rec

    def dyn_closure(self, t_idx, x_idx):
        rels = self.get_relations()
        z_input = self.drop(self.factors[t_idx])
        z_context = rels[x_idx].matmul(z_input).view(-1, self.nr * self.nz)
        z_gen = self.dynamic(z_context)
        return self.activation(z_gen)

    def generate(self, nsteps):
        z = self.factors[-1]
        z_gen = []
        for t in range(nsteps):
            z = self.update_z(z)
            z_gen.append(z)
        z_gen = torch.stack(z_gen)
        x_gen = self.decode_z(z_gen)
        return x_gen, z_gen

    def factors_parameters(self):
        yield self.factors

    def rel_parameters(self):
        assert self.mode is not None
        yield self.rel_weights
