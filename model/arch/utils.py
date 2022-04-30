import torch
import random
import numpy as np


def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i

    return mapped_label

def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)

    # torch cuda
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_hyperparameters(dataset):
    if dataset not in ['CUB', 'SUN', 'AWA1', 'AWA2']:
        print('hyperparameters error!')
        return exit(-1)
    # the basic hyperparameters
    hyperparameters = {
        'dataset': dataset,
        'num_shots': 0,
        'generalized': True,
        'device': 'cuda',
        'auxiliary_data_source': 'attributes',
        'latent_size': 64,
        'batch_size': 50,
        'lr_gen_model': 1.5e-04,
        'lr_iem_model': 3.3e-05,
        'lr_aux_cls': 7.4e-03,
        'epochs': 100,
        'loss': 'l1',
        'samples_per_class': {'CUB': (200, 0, 390, 0), 'SUN': (200, 0, 410, 0),
                              'AWA1': (200, 0, 460, 0), 'AWA2': (200, 0, 480, 0)},
        'lr_cls': 0.0005,
        'cls_batch_size': 32,
    }
    # the data-special hyperparameters
    hidden_size_rule = [
        {'dataset': 'CUB', 'hidden_size_rule': {'resnet_features': (1560, 1665), 'attributes': (1450, 665)}},
        {'dataset': 'SUN', 'hidden_size_rule': {'resnet_features': (1560, 1660), 'attributes': (1450, 665)}},
        {'dataset': 'AWA1', 'hidden_size_rule': {'resnet_features': (1560, 1680), 'attributes': (1450, 665)}},
        {'dataset': 'AWA2', 'hidden_size_rule': {'resnet_features': (1560, 1680), 'attributes': (1450, 665)}},
    ]
    warmup = [
        {'dataset': 'CUB', 'warmup': {'beta': {'factor': 2.555, 'end_epoch': 93, 'start_epoch': 0},
                                      'distance': {'factor': 8.21, 'end_epoch': 22, 'start_epoch': 7},
                                      'classify': {'factor': 310.5, 'end_epoch': 21, 'start_epoch': 0}}},
        {'dataset': 'SUN', 'warmup': {'beta': {'factor': 2.455, 'end_epoch': 93, 'start_epoch': 0},
                                      'distance': {'factor': 10.1, 'end_epoch': 22, 'start_epoch': 7},
                                      'classify': {'factor': 305.555, 'end_epoch': 22, 'start_epoch': 0}}},
        {'dataset': 'AWA1', 'warmup': {'beta': {'factor': 7./3., 'end_epoch': 93, 'start_epoch': 0},
                                      'distance': {'factor': 8.11, 'end_epoch': 23, 'start_epoch': 6},
                                      'classify': {'factor': 290.13, 'end_epoch': 16, 'start_epoch': 0}}},
        {'dataset': 'AWA2', 'warmup': {'beta': {'factor': 3.554, 'end_epoch': 93, 'start_epoch': 0},
                                      'distance': {'factor': 12.0, 'end_epoch': 22, 'start_epoch': 6}, #12
                                      'classify': {'factor': 100.5, 'end_epoch': 19, 'start_epoch': 0}}}, #80.5
    ]
    cls_train_steps = [
        {'dataset': 'CUB', 'num_shots': 0, 'generalized': True, 'cls_train_steps': 50},  # 30
        {'dataset': 'SUN', 'num_shots': 0, 'generalized': True, 'cls_train_steps': 50}, # 30
        {'dataset': 'AWA1', 'num_shots': 0, 'generalized': True, 'cls_train_steps': 50}, # 25
        {'dataset': 'AWA2', 'num_shots': 0, 'generalized': True, 'cls_train_steps': 300}, # 200
    ]
    num_classes = [
        {'dataset': 'CUB', 'num_classes': 200},
        {'dataset': 'SUN', 'num_classes': 717},
        {'dataset': 'AWA1', 'num_classes': 50},
        {'dataset': 'AWA2', 'num_classes': 50},
    ]
    # config by dataset
    hyperparameters['hidden_size_rule'] = [x['hidden_size_rule'] for x in hidden_size_rule
                                           if all([hyperparameters['dataset'] == x['dataset']])][0]
    hyperparameters['warmup'] = [x['warmup'] for x in warmup
                                 if all([hyperparameters['dataset'] == x['dataset']])][0]
    hyperparameters['cls_train_steps'] = [x['cls_train_steps'] for x in cls_train_steps
                                          if all([hyperparameters['dataset'] == x['dataset']])][0]
    hyperparameters['num_classes'] = [x['num_classes'] for x in num_classes
                                      if all([hyperparameters['dataset'] == x['dataset']])][0]

    return hyperparameters
