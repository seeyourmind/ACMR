import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.bias.data.fill_(0)
        nn.init.xavier_uniform_(m.weight, gain=0.5)

    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class encoder_template(nn.Module):
    def __init__(self, input_dim, latent_size, hidden_size_rule, device):
        super(encoder_template, self).__init__()

        if len(hidden_size_rule) == 2:
            self.layer_sizes = [input_dim, hidden_size_rule[0], latent_size]
        elif len(hidden_size_rule) == 3:
            self.layer_sizes = [input_dim, hidden_size_rule[0], hidden_size_rule[1], latent_size]

        modules = []
        for i in range(len(self.layer_sizes) - 2):
            modules.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1]))
            modules.append(nn.ReLU())

        self.feature_encoder = nn.Sequential(*modules)

        self._mu = nn.Linear(in_features=self.layer_sizes[-2], out_features=latent_size)
        self._logvar = nn.Linear(in_features=self.layer_sizes[-2], out_features=latent_size)

        self.apply(weights_init)

        self.to(device)

    def forward(self, x):
        h = self.feature_encoder(x).squeeze()

        mu = self._mu(h).squeeze()
        logvar = self._logvar(h).squeeze()

        return mu, logvar


class decoder_template(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_size_rule, device):
        super(decoder_template, self).__init__()

        self.layer_sizes = [input_dim, hidden_size_rule[-1], output_dim]

        self.feature_decoder = nn.Sequential(nn.Linear(input_dim, self.layer_sizes[1]), nn.ReLU(),
                                             nn.Linear(self.layer_sizes[1], output_dim))

        self.apply(weights_init)

        self.to(device)

    def forward(self, x):
        return self.feature_decoder(x).squeeze()


class MixInformation(nn.Module):
    def __init__(self, channel, hidden, device):
        super(MixInformation, self).__init__()
        self.nchannel = channel*2
        self.mix_net = nn.Sequential(nn.Linear(channel*2, hidden),
                                     nn.ReLU(),
                                     nn.Linear(hidden, channel),
                                     nn.Sigmoid())
        self.apply(weights_init)
        self.to(device)

    def forward(self, x, y):
        idx = torch.randperm(self.nchannel)
        mixV = torch.cat((x, y), dim=1).t()
        shuffleV = mixV[idx].t()
        W = self.mix_net(shuffleV)

        return W*x, W*y


class LINEAR_LOGSOFTMAX(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)
        self.logic = nn.LogSoftmax(dim=1)
        self.lossfunction = nn.NLLLoss()

    def forward(self, x):
        o = self.logic(self.fc(x))
        return o