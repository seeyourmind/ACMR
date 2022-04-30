import torch.optim as optim
import torch.nn as nn
import torch
import os
from arch.ACMR import Model
from arch.data_loader import DATA_LOADER as dataloader
import arch.final_classifier as ncls
import arch.models as models
import arch.utils as utils


def save_vae(model, data_name, save_root='save'):
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    torch.save(model, f'{save_root}/{data_name}-ACMR.pth')

def save_cls(model, data_name, save_root='save'):
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    torch.save(model, f'{save_root}/{data_name}-Classifier.pth.tar')

def sample_train_data(features, label, sample_per_class):
    sample_per_class = int(sample_per_class)

    if sample_per_class != 0 and len(label) != 0:

        classes = label.unique()

        for i, s in enumerate(classes):

            features_of_that_class = features[label == s, :]  # order of features and labels must coincide
            # if number of selected features is smaller than the number of features we want per class:
            multiplier = torch.ceil(torch.cuda.FloatTensor(
                [max(1, sample_per_class / features_of_that_class.size(0))])).long().item()

            features_of_that_class = features_of_that_class.repeat(multiplier, 1)

            if i == 0:
                features_to_return = features_of_that_class[:sample_per_class, :]
                labels_to_return = s.repeat(sample_per_class)
            else:
                features_to_return = torch.cat(
                    (features_to_return, features_of_that_class[:sample_per_class, :]), dim=0)
                labels_to_return = torch.cat((labels_to_return, s.repeat(sample_per_class)), dim=0)

        return features_to_return, labels_to_return
    else:
        return torch.cuda.FloatTensor([]), torch.cuda.LongTensor([])

def train_VAE_step(model, img, att, label, optimizer, criterion, warmup, c_epoch):
    shuffle_classification_criterion = criterion[1]
    criterion = criterion[0]

    res = model(img, att)
    img_from_img = res['rec']['img']
    att_from_att = res['rec']['att']
    mu_img = res['dis']['mui']
    mu_att = res['dis']['mua']
    logvar_img = res['dis']['logvari']
    logvar_att = res['dis']['logvara']
    out_v = res['cls']['vis']
    out_s = res['cls']['sem']

    # Loss_Rec
    reconstruction_loss = criterion(img_from_img, img) + criterion(att_from_att, att)

    # KL-Divergence
    KLD = (0.5 * torch.sum(1 + logvar_att - mu_att.pow(2) - logvar_att.exp())) \
          + (0.5 * torch.sum(1 + logvar_img - mu_img.pow(2) - logvar_img.exp()))

    # Loss_MA
    distance = torch.sqrt(torch.sum((mu_img - mu_att) ** 2, dim=1) + \
                          torch.sum((torch.sqrt(logvar_img.exp()) - torch.sqrt(logvar_att.exp())) ** 2, dim=1))
    distance = distance.sum()

    # Loss_Rep
    loss_v = shuffle_classification_criterion(out_v, label)
    loss_s = shuffle_classification_criterion(out_s, label)
    classification_loss = loss_v + loss_s

    # scale the loss terms according to the warmup schedule
    f1 = 1.0 * (c_epoch - warmup['beta']['start_epoch']) / (
                1.0 * (warmup['beta']['end_epoch'] - warmup['beta']['start_epoch']))
    f1 = f1 * (1.0 * warmup['beta']['factor'])
    beta = torch.cuda.FloatTensor([min(max(f1, 0), warmup['beta']['factor'])])

    f2 = 1.0 * (c_epoch - warmup['distance']['start_epoch']) / (
                1.0 * (warmup['distance']['end_epoch'] - warmup['distance']['start_epoch']))
    f2 = f2 * (1.0 * warmup['distance']['factor'])
    distance_factor = torch.cuda.FloatTensor([min(max(f2, 0), warmup['distance']['factor'])])

    f3 = 1.0 * (c_epoch - warmup['classify']['start_epoch']) / (
            1.0 * (warmup['classify']['end_epoch'] - warmup['classify']['start_epoch']))
    f3 = f3 * (1.0 * warmup['classify']['factor'])
    class_factor = torch.cuda.FloatTensor([min(max(f3, 0), warmup['classify']['factor'])])

    ##############################################
    # Put the loss together and call the optimizer
    ##############################################

    optimizer.zero_grad()

    loss = reconstruction_loss - beta * KLD

    if distance_factor > 0:
        loss += distance_factor * distance

    loss += class_factor * classification_loss

    loss.backward()

    optimizer.step()

    return loss.item(), distance.item(), classification_loss.item()

def train_vae(model, dataset, hypers):
    warmup = hypers['warmup']
    batch_size = hypers['batch_size']
    device = hypers['device']
    losses = []

    # An optimizer for all encoders and decoders is defined here
    parameters_to_optimize = list(model.parameters())
    for datatype in model.all_data_sources:
        parameters_to_optimize += list(model.encoder[datatype].parameters())
        parameters_to_optimize += list(model.decoder[datatype].parameters())

    parameters_sas = []
    for datatype in model.all_data_sources:
        parameters_sas += list(model.IEM[datatype].parameters())

    parameters_cls = []
    for datatype in model.all_data_sources:
        parameters_cls += list(model.aux_cls[datatype].parameters())

    optimizer = optim.Adam([{'params': parameters_to_optimize, 'lr': hypers['lr_gen_model']},
                            {'params': parameters_sas, 'lr': hypers['lr_iem_model']},
                            {'params': parameters_cls, 'lr': hypers['lr_aux_cls']}],
                           betas=(0.9, 0.999), weight_decay=0, eps=1e-08, amsgrad=True)

    reconstruction_criterion = nn.L1Loss(size_average=False)
    shuffle_classification_criterion = nn.NLLLoss()
    criterion = (reconstruction_criterion, shuffle_classification_criterion)


    # leave both statements
    model.train()
    model.reparameterize_with_noise = True

    uniq_label = dataset.seenclasses.long()

    print('train for reconstruction')
    for epoch in range(0, hypers['epochs']):
        current_epoch = epoch

        i = -1
        for iters in range(0, dataset.ntrain, batch_size):
            i += 1

            label, data_from_modalities = dataset.next_batch(batch_size)

            # label = utils.map_label(label.long(), uniq_label).to(device)
            label = label.long().to(device)
            for j in range(len(data_from_modalities)):
                data_from_modalities[j] = data_from_modalities[j].to(device)
                data_from_modalities[j].requires_grad = False

            loss, dis_loss, cls_loss = train_VAE_step(model, data_from_modalities[0], data_from_modalities[1], label, optimizer, criterion, warmup, current_epoch)

            # if i % 50 == 0:
                # print('epoch ' + str(epoch) + '\t| iter ' + str(i) +
                #       '\t| loss ' + str(loss)[:7] + ' | distance ' + str(dis_loss)[:5] + ' | classify ' + str(cls_loss)[:5])

            if i % 50 == 0 and i > 0:
                losses.append(loss)

    # turn into evaluation mode:
    model.eval()
    for key, value in model.encoder.items():
        model.encoder[key].eval()
    for key, value in model.decoder.items():
        model.decoder[key].eval()
    for key, value in model.aux_cls.items():
        model.aux_cls[key].eval()
    for key, value in model.IEM.items():
        model.IEM[key].eval()

    return losses

def train7test_classifier(model, classifier, dataset, hypers):
    def convert_to_z(features, encoder, aux):
        # aux=None
        if features.size(0) != 0:
            mu, logvar = encoder(features)
            mu_, logvar_ = aux(mu, logvar)
            z = model.reparameterize(mu_, logvar_)

            return z
        else:
            return torch.cuda.FloatTensor([])
    # read necessary config setting
    device = hypers['device']
    data_name = hypers['dataset']
    img_seen_samples = hypers['samples_per_class'][data_name][0]
    att_unseen_samples = hypers['samples_per_class'][data_name][2]
    lr_cls = hypers['lr_cls']
    classifier_batch_size = hypers['cls_batch_size']
    cls_train_epochs = hypers['cls_train_steps']
    history = []  # stores accuracies

    cls_seenclasses = dataset.seenclasses.long().cuda()
    cls_novelclasses = dataset.novelclasses.long().cuda()

    train_seen_feat = dataset.data['train_seen']['resnet_features']
    train_seen_label = dataset.data['train_seen']['labels']

    novelclass_aux_data = dataset.novelclass_aux_data
    novel_corresponding_labels = dataset.novelclasses.long().to(model.device)

    # The resnet_features for testing the classifier are loaded here
    novel_test_feat = dataset.data['test_unseen']['resnet_features']
    seen_test_feat = dataset.data['test_seen']['resnet_features']
    test_seen_label = dataset.data['test_seen']['labels']
    test_novel_label = dataset.data['test_unseen']['labels']

    print('mode: gzsl')
    with torch.no_grad():
        # get latent z of seen/unseen for testing classifier
        model.reparameterize_with_noise = False

        mu1, var1 = model.encoder['resnet_features'](novel_test_feat)
        mu1, var1 = model.IEM['resnet_features'](mu1, var1)
        test_novel_X = model.reparameterize(mu1, var1).to(device).data
        test_novel_Y = test_novel_label.to(device)

        mu2, var2 = model.encoder['resnet_features'](seen_test_feat)
        mu2, var2 = model.IEM['resnet_features'](mu2, var2)
        test_seen_X = model.reparameterize(mu2, var2).to(device).data
        test_seen_Y = test_seen_label.to(device)

        model.reparameterize_with_noise = True

        # sample training data
        img_seen_feat, img_seen_label = sample_train_data(train_seen_feat, train_seen_label, img_seen_samples)
        att_unseen_feat, att_unseen_label = sample_train_data(novelclass_aux_data, novel_corresponding_labels, att_unseen_samples)
        # covert original feature to latent feature
        z_seen_img = convert_to_z(img_seen_feat, model.encoder['resnet_features'], model.IEM['resnet_features'])
        z_unseen_att = convert_to_z(att_unseen_feat, model.encoder[model.auxiliary_data_source], model.IEM[model.auxiliary_data_source])

        train_Z = [z_seen_img, z_unseen_att]
        train_L = [img_seen_label, att_unseen_label]

        train_X = torch.cat(train_Z, dim=0)
        train_Y = torch.cat(train_L, dim=0)

    cls = ncls.CLASSIFIER(classifier, train_X, train_Y, test_seen_X, test_seen_Y, test_novel_X, test_novel_Y,
                          cls_seenclasses, cls_novelclasses, model.num_classes, device,
                          lr_cls, 0.3, 1, classifier_batch_size, True)

    for k in range(cls_train_epochs):
        if k > 0:
            cls.acc_seen, cls.acc_novel, cls.H = cls.fit()

            # print('[%.1f]\t novel=%.4f, seen=%.4f, h=%.4f , loss=%.4f' % (k, cls.acc_novel, cls.acc_seen, cls.H, cls.average_loss))
            history.append([torch.tensor(cls.acc_seen).item(), torch.tensor(cls.acc_novel).item(), torch.tensor(cls.H).item()])

    return torch.tensor(cls.acc_seen).item(), torch.tensor(cls.acc_novel).item(), torch.tensor(cls.H).item(), history

if __name__ == '__main__':
    import numpy as np
    # get model config
    data_name = 'AWA2'
    datap = 'data'
    hypers = utils.get_hyperparameters(data_name)

    # get dataloader
    dataset = dataloader(hypers['dataset'], hypers['auxiliary_data_source'], data_path=datap, device=hypers['device'])

    # init model
    utils.init_seeds()
    save_root = './save'
    model = Model(hypers)
    model.to(hypers['device'])

    train_vae(model, dataset, hypers)
    # save_vae(model, data_name)

    # init classifier
    classifier = models.LINEAR_LOGSOFTMAX(hypers['latent_size'], hypers['num_classes'])
    classifier.apply(models.weights_init)

    u, s, h, history = train7test_classifier(model, classifier, dataset, hypers)
    # save_cls(classifier, data_name)

    acc = [hi[2] for hi in history]
    print(f'[{data_name}][{acc.index(max(acc))}]: {max(acc)}')

