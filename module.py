import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, ninp, nhid, nout, nlayers, dropout):
        super(MLP, self).__init__()
        self.ninp = ninp
        # modules
        if nlayers == 1:
            self.module = nn.Linear(ninp, nout)
        else:
            modules = [nn.Linear(ninp, nhid), nn.ReLU(), nn.Dropout(dropout)]
            nlayers -= 1
            while nlayers > 1:
                modules += [nn.Linear(nhid, nhid), nn.ReLU(), nn.Dropout(dropout)]
                nlayers -= 1
            modules.append(nn.Linear(nhid, nout))
            self.module = nn.Sequential(*modules)

    def forward(self, input):
        return self.module(input)
