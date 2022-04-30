import torch

from arch.data_loader import DATA_LOADER as dataloader
import arch.utils as autils


def test_classifier(model, cls, dataset, hypers):
    # read necessary config setting
    device = hypers['device']

    cls_seenclasses = dataset.seenclasses.long().cuda()
    cls_novelclasses = dataset.novelclasses.long().cuda()

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

    acc_seen = cls.val_gzsl(test_seen_X, test_seen_Y, cls_seenclasses)
    acc_novel = cls.val_gzsl(test_novel_X, test_novel_Y, cls_novelclasses)
    if (acc_seen + acc_novel) > 0:
        H = (2 * acc_seen * acc_novel) / (acc_seen + acc_novel)
    else:
        H = 0
    print('GZSL: novel=%0.4f, seen=%.4f, h=%.4f' % (acc_novel, acc_seen, H))


if __name__ == '__main__':
    # get model config
    data_name = 'CUB'
    hypers = autils.get_hyperparameters(data_name)

    # get dataloader
    dataset = dataloader(hypers['dataset'], hypers['auxiliary_data_source'], device=hypers['device'])

    autils.init_seeds()
    # init model
    model = torch.load(f'save/{data_name}-ACMR.pt')
    model.eval()
    # init classifier
    fcls = torch.load(f'save/{data_name}-GZSC_classifier.pt')

    test_classifier(model, fcls, dataset, hypers)


