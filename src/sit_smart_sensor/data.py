from collections import Counter

import numpy as np
import torch
from torchvision import transforms, datasets


def get_train_transforms(size, rotation=0, brightness=0., contrast=0., saturation=0., hue=0.,
                         random_gray_scale=0., **kwargs):
    return transforms.Compose([transforms.RandomRotation(rotation)
                                  , transforms.Resize(size)
                                  , transforms.RandomHorizontalFlip(),
                               transforms.ColorJitter(brightness=brightness, contrast=contrast,
                                                      saturation=saturation, hue=hue),
                               transforms.RandomGrayscale(p=random_gray_scale),
                               transforms.ToTensor()])


def get_val_transforms(size, **kwargs):
    return transforms.Compose([transforms.Resize(size),
                               transforms.ToTensor()])


def get_test_transforms(size, **kwargs):
    return transforms.Compose([transforms.Resize(size),
                               transforms.ToTensor()])


def get_dataset(root,size, ratio_train=0.8, seed=13, print_summary=True, train_transforms=None):
    if ratio_train <= 0 or ratio_train >= 1:
        raise ValueError(f"ratio_train must be between 0 and 1. But it is {ratio_train}")
    if train_transforms is None:
        train_transforms = {}
    transform_train = get_train_transforms(**train_transforms, size=size)
    transform_val = get_val_transforms(size)
    dataset = datasets.ImageFolder(root=root,
                                   transform=transform_train)
    dataset_val = datasets.ImageFolder(root=root,
                                       transform=transform_val)
    if len(dataset) == 0:
        raise ValueError(f"Dataset is empty. Please check the path {root}")

    # split train and validation
    rng = np.random.default_rng(seed)
    train_ind = rng.choice(len(dataset), size=int(len(dataset) * ratio_train), replace=False)
    val_ind = np.setdiff1d(np.arange(len(dataset)), train_ind)

    # summary
    train_summary = Counter(np.array(dataset.targets)[train_ind])
    val_summay = Counter(np.array(dataset.targets)[val_ind])
    assert len(train_summary) == len(val_summay), "Train and validation datasets have different number of classes. Got " \
                                                  f"{len(train_summary)} and {len(val_summay)} classes respectively."
    if print_summary:
        print("Train set summary: ")
        print(' Number of samples:', len(train_ind))
        print(' Number of classes:', len(train_summary))
        print(' Class distribution:')
        for k in sorted(train_summary.keys()):
            print(f'  {k}: {train_summary[k]}')
        print()
        print("Validation set summary: ")
        print(' Number of samples:', len(val_ind))
        print(' Number of classes:', len(val_summay))
        print(' Class distribution:')
        for k in sorted(val_summay.keys()):
            print(f'  {k}: {val_summay[k]}')

    train_dataset = torch.utils.data.Subset(dataset, train_ind)
    val_dataset = torch.utils.data.Subset(dataset_val, val_ind)

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError(f"Train or validation dataset is empty. Please check the path {root}")

    return train_dataset, val_dataset
