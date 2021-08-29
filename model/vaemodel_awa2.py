# vaemodel + sas + cot
import copy
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.utils.data import Dataset, DataLoader
from data_loader import DATA_LOADER as dataloader
import final_classifier as classifier
import models


class LINEAR_LOGSOFTMAX(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)
        self.logic = nn.LogSoftmax(dim=1)
        self.lossfunction = nn.NLLLoss()

    def forward(self, x):
        o = self.logic(self.fc(x))
        return o


class Model(nn.Module):

    def __init__(self, hyperparameters):
        super(Model, self).__init__()

        self.device = hyperparameters['device']
        self.auxiliary_data_source = hyperparameters['auxiliary_data_source']
        self.all_data_sources = ['resnet_features', self.auxiliary_data_source]
        self.DATASET = hyperparameters['dataset']
        self.num_shots = hyperparameters['num_shots']
        self.latent_size = hyperparameters['latent_size']
        self.batch_size = hyperparameters['batch_size']
        self.hidden_size_rule = hyperparameters['hidden_size_rule']
        self.warmup = hyperparameters['model_specifics']['warmup']
        self.generalized = hyperparameters['generalized']
        self.classifier_batch_size = 32
        self.img_seen_samples = hyperparameters['samples_per_class'][self.DATASET][0]
        self.att_seen_samples = hyperparameters['samples_per_class'][self.DATASET][1]
        self.att_unseen_samples = hyperparameters['samples_per_class'][self.DATASET][2]
        self.img_unseen_samples = hyperparameters['samples_per_class'][self.DATASET][3]
        self.reco_loss_function = hyperparameters['loss']
        self.nepoch = hyperparameters['epochs']
        self.lr_cls = hyperparameters['lr_cls']
        self.cross_reconstruction = hyperparameters['model_specifics']['cross_reconstruction']
        self.cls_train_epochs = hyperparameters['cls_train_steps']
        self.zae_train_epochs = hyperparameters['zae_train_steps']
        self.dataset = dataloader(self.DATASET, copy.deepcopy(self.auxiliary_data_source), device=self.device)

        if self.DATASET == 'CUB':
            self.num_classes = 200
            self.num_novel_classes = 50
        elif self.DATASET == 'SUN':
            self.num_classes = 717
            self.num_novel_classes = 72
        elif self.DATASET == 'AWA1' or self.DATASET == 'AWA2':
            self.num_classes = 50
            self.num_novel_classes = 10

        feature_dimensions = [2048, self.dataset.aux_data.size(1)]

        # Here, the encoders and decoders for all modalities are created and put into dict

        self.encoder = {}

        for datatype, dim in zip(self.all_data_sources, feature_dimensions):
            self.encoder[datatype] = models.encoder_template(dim, self.latent_size, self.hidden_size_rule[datatype], self.device)
            print(str(datatype) + ' ' + str(dim))

        self.decoder = {}
        for datatype, dim in zip(self.all_data_sources, feature_dimensions):
            self.decoder[datatype] = models.decoder_template(self.latent_size, dim, self.hidden_size_rule[datatype], self.device)

        self.aux_cls = {}
        for datatype in self.all_data_sources:
            self.aux_cls[datatype] = LINEAR_LOGSOFTMAX(self.latent_size, self.num_classes).to(self.device)
            self.aux_cls[datatype].apply(models.weights_init)
        self.shuffle_classification_criterion = nn.NLLLoss()

        self.sas = {}
        for datatype in self.all_data_sources:
            self.sas[datatype] = models.InformationEnhancement(self.latent_size, 99, self.device)

        self.zae = {}
        for datatype in ['mu', 'sigma']:
            self.zae[datatype] = models.ZAutoEncoder(self.latent_size, 50, 127, self.device)

        # An optimizer for all encoders and decoders is defined here
        parameters_to_optimize = list(self.parameters())
        for datatype in self.all_data_sources:
            parameters_to_optimize += list(self.encoder[datatype].parameters())
            parameters_to_optimize += list(self.decoder[datatype].parameters())

        parameters_sas = []
        for datatype in self.all_data_sources:
            parameters_sas += list(self.sas[datatype].parameters())

        parameters_cls = []
        for datatype in self.all_data_sources:
            parameters_cls += list(self.aux_cls[datatype].parameters())

        parameters_zae = []
        for datatype in ['mu', 'sigma']:
            parameters_zae += list(self.zae[datatype].parameters())

        # hyperparameters['lr_gen_model']
        self.optimizer = optim.Adam([{'params': parameters_to_optimize, 'lr': hyperparameters['lr_gen_model']},
                                     {'params': parameters_sas, 'lr': 3.3e-05},
                                     {'params': parameters_cls, 'lr': 7.4e-03}],
                                    betas=(0.9, 0.999), weight_decay=0, eps=1e-08, amsgrad=True)
        self.zae_optimizer = optim.Adam(parameters_zae, lr=0.00025, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)

        if self.reco_loss_function == 'l2':
            self.reconstruction_criterion = nn.MSELoss(size_average=False)

        elif self.reco_loss_function == 'l1':
            self.reconstruction_criterion = nn.L1Loss(size_average=False)

    def reparameterize(self, mu, logvar):
        if self.reparameterize_with_noise:
            sigma = torch.exp(logvar)
            eps = torch.cuda.FloatTensor(logvar.size()[0], 1).normal_(0, 1)
            eps = eps.expand(sigma.size())
            return mu + sigma * eps
        else:
            return mu

    def forward(self):
        pass

    def map_label(self, label, classes):
        mapped_label = torch.LongTensor(label.size()).to(self.device)
        for i in range(classes.size(0)):
            mapped_label[label == classes[i]] = i

        return mapped_label

    def add_cross_weight(self, img, att):
        Q = img.view(-1, self.latent_size, 1)
        K = att.view(-1, 1, self.latent_size)
        R = torch.bmm(Q, K)
        soft_R = F.softmax(R, 1)
        _img = torch.bmm(soft_R, img.unsqueeze(2)).squeeze()
        _att = torch.bmm(soft_R, att.unsqueeze(2)).squeeze()

        return _img, _att

    def trainstep(self, img, att, label):

        ##############################################
        # Encode image features and additional
        # features
        ##############################################

        mu_img, logvar_img = self.encoder['resnet_features'](img)
        mu_img, logvar_img = self.sas['resnet_features'](mu_img, logvar_img)
        z_from_img = self.reparameterize(mu_img, logvar_img)

        mu_att, logvar_att = self.encoder[self.auxiliary_data_source](att)
        mu_att, logvar_att = self.sas[self.auxiliary_data_source](mu_att, logvar_att)
        z_from_att = self.reparameterize(mu_att, logvar_att)

        ##############################################
        # Reconstruct inputs
        ##############################################

        img_from_img = self.decoder['resnet_features'](z_from_img)
        att_from_att = self.decoder[self.auxiliary_data_source](z_from_att)

        reconstruction_loss = self.reconstruction_criterion(img_from_img, img) \
                              + self.reconstruction_criterion(att_from_att, att)

        z_from_img_, z_from_att_ = self.add_cross_weight(z_from_img, z_from_att)
        out_v = self.aux_cls['resnet_features'](z_from_img_)
        out_s = self.aux_cls['attributes'](z_from_att_)

        ##############################################
        # KL-Divergence
        ##############################################

        KLD = (0.5 * torch.sum(1 + logvar_att - mu_att.pow(2) - logvar_att.exp())) \
              + (0.5 * torch.sum(1 + logvar_img - mu_img.pow(2) - logvar_img.exp()))

        ##############################################
        # Distribution Alignment
        ##############################################
        distance = torch.sqrt(torch.sum((mu_img - mu_att) ** 2, dim=1) +\
                              torch.sum((torch.sqrt(logvar_img.exp()) - torch.sqrt(logvar_att.exp())) ** 2, dim=1))

        distance = distance.sum()

        loss_v = self.shuffle_classification_criterion(out_v, label)
        loss_s = self.shuffle_classification_criterion(out_s, label)
        classification_loss = loss_v + loss_s

        ##############################################
        # scale the loss terms according to the warmup
        # schedule
        ##############################################
        
        f2 = 1.0 * (self.current_epoch - self.warmup['beta']['start_epoch']) / (
                    1.0 * (self.warmup['beta']['end_epoch'] - self.warmup['beta']['start_epoch']))
        f2 = f2 * (1.0 * self.warmup['beta']['factor'])
        beta = torch.cuda.FloatTensor([min(max(f2, 0), self.warmup['beta']['factor'])])

        f3 = 1.0 * (self.current_epoch - self.warmup['distance']['start_epoch']) / (
                    1.0 * (self.warmup['distance']['end_epoch'] - self.warmup['distance']['start_epoch']))
        f3 = f3 * (1.0 * self.warmup['distance']['factor'])
        distance_factor = torch.cuda.FloatTensor([min(max(f3, 0), self.warmup['distance']['factor'])])

        f4 = 1.0 * (self.current_epoch - self.warmup['classify']['start_epoch']) / (
                1.0 * (self.warmup['classify']['end_epoch'] - self.warmup['classify']['start_epoch']))
        f4 = f4 * (1.0 * self.warmup['classify']['factor'])
        class_factor = torch.cuda.FloatTensor([min(max(f4, 0), self.warmup['classify']['factor'])])

        ##############################################
        # Put the loss together and call the optimizer
        ##############################################

        self.optimizer.zero_grad()

        loss = reconstruction_loss - beta * KLD

        if distance_factor > 0:
            loss += distance_factor * distance

        loss += class_factor * classification_loss

        loss.backward()

        self.optimizer.step()

        return loss.item(), distance.item(), classification_loss.item()

    def train_vae(self):

        losses = []

        # leave both statements
        self.train()
        self.reparameterize_with_noise = True

        print('train for reconstruction')
        for epoch in range(0, self.nepoch):
            self.current_epoch = epoch

            i = -1
            for iters in range(0, self.dataset.ntrain, self.batch_size):
                i += 1

                label, data_from_modalities = self.dataset.next_batch(self.batch_size)

                label = label.long().to(self.device)
                for j in range(len(data_from_modalities)):
                    data_from_modalities[j] = data_from_modalities[j].to(self.device)
                    data_from_modalities[j].requires_grad = False

                loss, dis_loss, cls_loss = self.trainstep(data_from_modalities[0], data_from_modalities[1], label)

                if i % 50 == 0:
                    print('epoch ' + str(epoch) + '\t| iter ' + str(i) +
                          '\t| loss ' + str(loss)[:7] + ' | distance ' + str(dis_loss)[:5] + ' | classify ' + str(cls_loss)[:5])

                if i % 50 == 0 and i > 0:
                    losses.append(loss)

        # turn into evaluation mode:
        for key, value in self.encoder.items():
            self.encoder[key].eval()
        for key, value in self.decoder.items():
            self.decoder[key].eval()
        for key, value in self.aux_cls.items():
            self.aux_cls[key].eval()
        for key, value in self.sas.items():
            self.sas[key].eval()

        return losses

    def sas_step(self, img, att):
        mu_img, logvar_img = self.encoder['resnet_features'](img)
        mu_att, logvar_att = self.encoder[self.auxiliary_data_source](att)

        mu_img_from_ma = self.zae['mu'](mu_att)
        sigma_img_from_sa = self.zae['sigma'](logvar_att)

        loss = self.reconstruction_criterion(mu_img_from_ma, mu_img) \
               + self.reconstruction_criterion(sigma_img_from_sa, logvar_img)

        self.zae_optimizer.zero_grad()
        loss.backward()
        self.zae_optimizer.step()

        return loss.item()

    def train_zae(self):

        losses = []

        train_img = self.dataset.data['train_seen']['resnet_features']
        train_att = self.dataset.data['train_seen'][self.auxiliary_data_source]
        train_lab = self.dataset.data['train_seen']['labels']
        uniqueLab = torch.unique(train_lab)
        print(f'{train_img.shape}, {train_att.shape}, {train_lab.shape}, {uniqueLab.shape}')
        print('处理训练集')

        def _get_mean_feat_map(old_pair, labels):
            re_lab, re_img, re_att = [], [], []
            loop_pair = list(old_pair)
            count = 0
            for i in tqdm(labels):
                re_lab.append(i)
                temp_i, temp_a = [], []
                for key, value_i, value_a in loop_pair:
                    if key == i:
                        temp_i.append(value_i.unsqueeze(0))
                        temp_a.append(value_a.unsqueeze(0))
                        count += 1
                if len(temp_i) > 0:
                    temp_i = torch.cat(temp_i, dim=0).mean(dim=0).unsqueeze(0)
                    temp_a = torch.cat(temp_a, dim=0).mean(dim=0).unsqueeze(0)
                    re_img.append(temp_i)
                    re_att.append(temp_a)
                else:
                    print(f'{i}: 没有样本')
            re_lab = torch.tensor(re_lab)
            re_img = torch.cat(re_img, dim=0)
            re_att = torch.cat(re_att, dim=0)
            # print(f'{re_lab.dtype}, {re_att.shape}, {re_img.shape}, {count}')
            return {'labels': re_lab, 'imgs': re_img, 'atts': re_att}
        dataset = _get_mean_feat_map(zip(train_lab, train_img, train_att), uniqueLab)

        def _sample_from_mean_feat(dataset, nsample=100):
            sample_per_class = int(nsample)
            if sample_per_class != 0:
                labels, img_feat, att_feat = dataset['labels'], dataset['imgs'], dataset['atts']
                for i, l in enumerate(labels):
                    img_of_that_class = img_feat[labels == l, :]
                    att_of_that_class = att_feat[labels == l, :]
                    multiplier = torch.ceil(torch.cuda.FloatTensor([max(1, sample_per_class/img_of_that_class.size(0))])).long().item()
                    img_of_that_class = img_of_that_class.repeat(multiplier, 1)
                    att_of_that_class = att_of_that_class.repeat(multiplier, 1)
                    if i == 0:
                        img_to_return = img_of_that_class[:sample_per_class, :]
                        att_to_return = att_of_that_class[:sample_per_class, :]
                        lab_to_return = l.repeat(sample_per_class)
                    else:
                        img_to_return = torch.cat((img_to_return, img_of_that_class[:sample_per_class, :]), dim=0)
                        att_to_return = torch.cat((att_to_return, att_of_that_class[:sample_per_class, :]), dim=0)
                        lab_to_return = torch.cat((lab_to_return, l.repeat(sample_per_class)), dim=0)
                return img_to_return, att_to_return, lab_to_return
            else:
                return torch.cuda.FloatTensor([]), torch.cuda.FloatTensor([]), torch.cuda.LongTensor([])
        train_img, train_att, train_lab = _sample_from_mean_feat(dataset)
        print(f'{train_img.shape}, {train_att.shape}, {train_lab.shape}')

        class TrainDataset(Dataset):
            """Face Landmarks dataset."""
            def __init__(self, train_X1, train_X2, train_Y):
                self.train_X1 = train_X1
                self.train_X2 = train_X2
                self.train_Y = train_Y.long()

            def __len__(self):
                return self.train_Y.size(0)

            def __getitem__(self, idx):
                return {'img': self.train_X1[idx, :], 'att': self.train_X2[idx, :], 'lab': self.train_Y[idx]}
        dataloader = DataLoader(TrainDataset(train_img, train_att, train_lab), batch_size=self.batch_size, shuffle=True, drop_last=True)

        # leave both statements
        self.train()
        self.reparameterize_with_noise = True

        # freeze encoder, decoder, aux, sas
        for key, value in self.encoder.items():
            for para in self.encoder[key].parameters():
                para.requires_grad = False
        for key, value in self.decoder.items():
            for para in self.decoder[key].parameters():
                para.requires_grad = False
        for key, value in self.aux_cls.items():
            for para in self.aux_cls[key].parameters():
                para.requires_grad = False
        for key, value in self.sas.items():
            for para in self.sas[key].parameters():
                para.requires_grad = False

        print('train for reconstruction')
        for epoch in range(0, self.zae_train_epochs):
            self.current_epoch = epoch

            i = -1
            for batch in dataloader:
                i += 1
                loss = self.sas_step(batch['img'], batch['att'])

                if i % 50 == 0:
                    print('epoch ' + str(epoch) + '\t| iter ' + str(i) + '\t| loss ' + str(loss)[:7])

                if i % 50 == 0 and i > 0:
                    losses.append(loss)

        # turn into evaluation mode:
        for key, value in self.zae.items():
            self.zae[key].eval()

        return losses

    def train_classifier(self, show_plots=False):

        if self.num_shots > 0:
            print('================  transfer features from test to train ==================')
            self.dataset.transfer_features(self.num_shots, num_queries='num_features')

        history = []  # stores accuracies

        cls_seenclasses = self.dataset.seenclasses.long().cuda()
        cls_novelclasses = self.dataset.novelclasses.long().cuda()

        train_seen_feat = self.dataset.data['train_seen']['resnet_features']
        train_seen_label = self.dataset.data['train_seen']['labels']

        novelclass_aux_data = self.dataset.novelclass_aux_data  # access as novelclass_aux_data['resnet_features'], novelclass_aux_data['attributes']
        seenclass_aux_data = self.dataset.seenclass_aux_data

        novel_corresponding_labels = self.dataset.novelclasses.long().to(self.device)
        seen_corresponding_labels = self.dataset.seenclasses.long().to(self.device)

        # The resnet_features for testing the classifier are loaded here
        novel_test_feat = self.dataset.data['test_unseen']['resnet_features']
        # self.dataset.test_novel_feature.to(self.device)
        seen_test_feat = self.dataset.data['test_seen']['resnet_features']
        # self.dataset.test_seen_feature.to(self.device)
        test_seen_label = self.dataset.data['test_seen']['labels']
        # self.dataset.test_seen_label.to(self.device)
        test_novel_label = self.dataset.data['test_unseen']['labels']
        # self.dataset.test_novel_label.to(self.device)

        train_unseen_feat = self.dataset.data['train_unseen']['resnet_features']
        train_unseen_label = self.dataset.data['train_unseen']['labels']

        # in ZSL mode:
        if self.generalized == False:
            # there are only 50 classes in ZSL (for CUB)
            # novel_corresponding_labels =list of all novel classes (as tensor)
            # test_novel_label = mapped to 0-49 in classifier function
            # those are used as targets, they have to be mapped to 0-49 right here:

            novel_corresponding_labels = self.map_label(novel_corresponding_labels, novel_corresponding_labels)

            if self.num_shots > 0:
                # not generalized and at least 1 shot means normal FSL setting (use only unseen classes)
                train_unseen_label = self.map_label(train_unseen_label, cls_novelclasses)

            # for FSL, we train_seen contains the unseen class examples
            # for ZSL, train seen label is not used
            # if self.num_shots>0:
            #    train_seen_label = self.map_label(train_seen_label,cls_novelclasses)

            test_novel_label = self.map_label(test_novel_label, cls_novelclasses)

            # map cls novelclasses last
            cls_novelclasses = self.map_label(cls_novelclasses, cls_novelclasses)

        if self.generalized:
            print('mode: gzsl')
            clf = LINEAR_LOGSOFTMAX(self.latent_size, self.num_classes)
        else:
            print('mode: zsl')
            clf = LINEAR_LOGSOFTMAX(self.latent_size, self.num_novel_classes)

        clf.apply(models.weights_init)

        with torch.no_grad():

            ####################################
            # preparing the test set
            # convert raw test data into z vectors
            ####################################

            self.reparameterize_with_noise = False

            mu1, var1 = self.encoder['resnet_features'](novel_test_feat)
            mu1, var1 = self.sas['resnet_features'](mu1, var1)
            test_novel_X = self.reparameterize(mu1, var1).to(self.device).data
            # test_novel_label = self.map_label(test_novel_label, cls_novelclasses)
            test_novel_Y = test_novel_label.to(self.device)

            mu2, var2 = self.encoder['resnet_features'](seen_test_feat)
            mu2, var2 = self.sas['resnet_features'](mu2, var2)
            test_seen_X = self.reparameterize(mu2, var2).to(self.device).data
            # test_seen_label = self.map_label(test_seen_label, cls_seenclasses)
            test_seen_Y = test_seen_label.to(self.device)

            ####################################
            # preparing the train set:
            # chose n random image features per
            # class. If n exceeds the number of
            # image features per class, duplicate
            # some. Next, convert them to
            # latent z features.
            ####################################

            self.reparameterize_with_noise = True

            def sample_train_data_on_sample_per_class_basis(features, label, sample_per_class):
                sample_per_class = int(sample_per_class)

                if sample_per_class != 0 and len(label) != 0:

                    classes = label.unique()

                    for i, s in enumerate(classes):

                        features_of_that_class = features[label == s, :]  # order of features and labels must coincide
                        # if number of selected features is smaller than the number of features we want per class:
                        multiplier = torch.ceil(torch.cuda.FloatTensor([max(1, sample_per_class / features_of_that_class.size(0))])).long().item()

                        features_of_that_class = features_of_that_class.repeat(multiplier, 1)

                        if i == 0:
                            features_to_return = features_of_that_class[:sample_per_class, :]
                            labels_to_return = s.repeat(sample_per_class)
                        else:
                            features_to_return = torch.cat((features_to_return, features_of_that_class[:sample_per_class, :]), dim=0)
                            labels_to_return = torch.cat((labels_to_return, s.repeat(sample_per_class)), dim=0)

                    return features_to_return, labels_to_return
                else:
                    return torch.cuda.FloatTensor([]), torch.cuda.LongTensor([])

            # some of the following might be empty tensors if the specified number of
            # samples is zero :

            img_seen_feat, img_seen_label = sample_train_data_on_sample_per_class_basis(train_seen_feat, train_seen_label, self.img_seen_samples)

            att_unseen_feat, att_unseen_label = sample_train_data_on_sample_per_class_basis(novelclass_aux_data, novel_corresponding_labels, self.att_unseen_samples)

            def convert_datapoints_to_z(features, encoder, aux=None, details=False):
                if features.size(0) != 0:
                    mu, logvar = encoder(features)
                    if aux is not None:
                        mu_, logvar_ = aux(mu, logvar)
                    z = self.reparameterize(mu_, logvar_)
                    if details is True:
                        mu_i = self.zae['mu'](mu)
                        logvar_i = self.zae['sigma'](logvar)
                        if aux is not None:
                            mu_i, logvar_i = aux(mu_i, logvar_i)
                            z_i = self.reparameterize(mu_, logvar_i)
                        else:
                            z_i = self.reparameterize(mu, logvar_i)
                        return z_i
                    else:
                        return z
                else:
                    return torch.cuda.FloatTensor([])

            z_seen_img = convert_datapoints_to_z(img_seen_feat, self.encoder['resnet_features'], self.sas['resnet_features'])
            z_unseen_att = convert_datapoints_to_z(att_unseen_feat, self.encoder[self.auxiliary_data_source], self.sas[self.auxiliary_data_source], True)
            
            train_Z = [z_seen_img, z_unseen_att]
            train_L = [img_seen_label, att_unseen_label]

            train_X = torch.cat(train_Z, dim=0)
            train_Y = torch.cat(train_L, dim=0)

        ############################################################
        ##### initializing the classifier and train one epoch
        ############################################################

        cls = classifier.CLASSIFIER(clf, train_X, train_Y, test_seen_X, test_seen_Y, test_novel_X,
                                    test_novel_Y,
                                    cls_seenclasses, cls_novelclasses,
                                    self.num_classes, self.device, self.lr_cls, 0.3, 1,
                                    self.classifier_batch_size,
                                    self.generalized)

        for k in range(self.cls_train_epochs):
            if k > 0:
                if self.generalized:
                    cls.acc_seen, cls.acc_novel, cls.H = cls.fit()
                else:
                    cls.acc = cls.fit_zsl()

            if self.generalized:

                print('[%.1f]     novel=%.4f, seen=%.4f, h=%.4f , loss=%.4f' % (
                    k, cls.acc_novel, cls.acc_seen, cls.H, cls.average_loss))

                history.append([torch.tensor(cls.acc_seen).item(), torch.tensor(cls.acc_novel).item(),
                                torch.tensor(cls.H).item()])

            else:
                print('[%.1f]  acc=%.4f ' % (k, cls.acc))
                history.append([0, torch.tensor(cls.acc).item(), 0])

        if self.generalized:
            return torch.tensor(cls.acc_seen).item(), torch.tensor(cls.acc_novel).item(), torch.tensor(
                cls.H).item(), history
        else:
            return 0, torch.tensor(cls.acc).item(), 0, history
