import torch
import torch.nn as nn
import torch.nn.functional as F
import arch.models as models


class Model(nn.Module):
    def __init__(self, hyperparameters):
        super(Model, self).__init__()

        self.device = hyperparameters['device']
        self.auxiliary_data_source = hyperparameters['auxiliary_data_source']
        self.all_data_sources = ['resnet_features', self.auxiliary_data_source]
        self.DATASET = hyperparameters['dataset']
        self.latent_size = hyperparameters['latent_size']
        self.hidden_size_rule = hyperparameters['hidden_size_rule']
        self.generalized = hyperparameters['generalized']
        self.reparameterize_with_noise = True

        if self.DATASET == 'CUB':
            self.num_classes = 200
            self.num_novel_classes = 50
            self.aux_data_size = 312
        elif self.DATASET == 'SUN':
            self.num_classes = 717
            self.num_novel_classes = 72
            self.aux_data_size = 102
        elif self.DATASET == 'AWA1' or self.DATASET == 'AWA2':
            self.num_classes = 50
            self.num_novel_classes = 10
            self.aux_data_size = 85

        feature_dimensions = [2048, self.aux_data_size]

        # Here, the encoders and decoders for all modalities are created and put into dict
        self.encoder = {}
        for datatype, dim in zip(self.all_data_sources, feature_dimensions):
            self.encoder[datatype] = models.encoder_template(dim, self.latent_size, self.hidden_size_rule[datatype], self.device)

        self.decoder = {}
        for datatype, dim in zip(self.all_data_sources, feature_dimensions):
            self.decoder[datatype] = models.decoder_template(self.latent_size, dim, self.hidden_size_rule[datatype], self.device)

        self.aux_cls = {}
        for datatype in self.all_data_sources:
            self.aux_cls[datatype] = models.LINEAR_LOGSOFTMAX(self.latent_size, self.num_classes).to(self.device)
            self.aux_cls[datatype].apply(models.weights_init)

        self.IEM = {}
        for datatype in self.all_data_sources:
            self.IEM[datatype] = models.MixInformation(self.latent_size, 99, self.device)

    def reparameterize(self, mu, logvar):
        if self.reparameterize_with_noise:
            sigma = torch.exp(logvar)
            eps = torch.cuda.FloatTensor(logvar.size()[0], 1).normal_(0, 1)
            eps = eps.expand(sigma.size())
            return mu + sigma * eps
        else:
            return mu

    def add_cross_weight(self, img, att):
        Q = img.view(-1, self.latent_size, 1)
        K = att.view(-1, 1, self.latent_size)
        R = torch.bmm(Q, K)
        soft_R = F.softmax(R, 1)
        _img = torch.bmm(soft_R, img.unsqueeze(2)).squeeze()
        _att = torch.bmm(soft_R, att.unsqueeze(2)).squeeze()

        return _img, _att

    def get_z_with_iem(self, inp, dtype='v', gener=True):
        self.reparameterize_with_noise = gener
        if dtype == 'v':
            mu, logvar = self.encoder['resnet_features'](inp)
            mu_, logvar_ = self.IEM['resnet_features'](mu, logvar)
        elif dtype == 's':
            mu, logvar = self.encoder[self.auxiliary_data_source](inp)
            mu_, logvar_ = self.IEM[self.auxiliary_data_source](mu, logvar)
        else:
            print('error data type! ')
            return -1
        z = self.reparameterize(mu_, logvar_)
        return z

    def forward(self, img, att):
        # encoder inference
        mu_img, logvar_img = self.encoder['resnet_features'](img)
        mu_img, logvar_img = self.IEM['resnet_features'](mu_img, logvar_img)
        z_from_img = self.reparameterize(mu_img, logvar_img)

        mu_att, logvar_att = self.encoder[self.auxiliary_data_source](att)
        mu_att, logvar_att = self.IEM[self.auxiliary_data_source](mu_att, logvar_att)
        z_from_att = self.reparameterize(mu_att, logvar_att)

        # decoder inference
        img_from_img = self.decoder['resnet_features'](z_from_img)
        att_from_att = self.decoder[self.auxiliary_data_source](z_from_att)

        # VSA inference
        z_from_img_, z_from_att_ = self.add_cross_weight(z_from_img, z_from_att)
        out_v = self.aux_cls['resnet_features'](z_from_img_)
        out_s = self.aux_cls['attributes'](z_from_att_)

        res = {'rec':  # reconstruction
                   {'img': img_from_img,
                    'att': att_from_att,
                    },
               'dis':  # distribution
                   {'mui': mu_img,
                    'mua': mu_att,
                    'logvari': logvar_img,
                    'logvara': logvar_att,
                    },
               'cls':
                   {'vis': out_v,
                    'sem': out_s}
               }
        return res

