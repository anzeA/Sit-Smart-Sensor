import numpy as np
import torch
from torchvision import transforms, datasets


def get_dummy_dataset(n_train, n_val):
    train_dataset = torch.utils.data.TensorDataset(torch.rand(n_train, 3, 224, 224),
                                                   torch.randint(0, 2, (n_train,)).float())
    val_dataset = torch.utils.data.TensorDataset(torch.rand(n_val, 3, 224, 224), torch.randint(0, 2, (n_val,)).float())
    return train_dataset, val_dataset


def get_train_transforms(size=(224, 224), rotation=10, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, **kwargs):
    return transforms.Compose([transforms.RandomRotation(rotation)
                                  , transforms.Resize(size)
                                  , transforms.RandomHorizontalFlip(),
                               transforms.ColorJitter(brightness=brightness, contrast=contrast,
                                                      saturation=saturation, hue=hue),
                               transforms.ToTensor()])


def get_val_transforms(size=(224, 224), **kwargs):
    return transforms.Compose([transforms.Resize(size),
                               transforms.ToTensor()])


def get_test_transforms(size=(224, 224), **kwargs):
    return transforms.Compose([transforms.Resize(size),
                               transforms.ToTensor()])


def get_dataset(root, ratio_train=0.8, seed=13, **kwargs):
    if ratio_train <= 0 or ratio_train >= 1:
        raise ValueError(f"ratio_train must be between 0 and 1. But it is {ratio_train}")
    transform_train = get_train_transforms(**kwargs)
    transform_val = get_val_transforms(**kwargs)
    dataset = datasets.ImageFolder(root=root,
                                   transform=transform_train)
    dataset_val = datasets.ImageFolder(root=root,
                                       transform=transform_val)
    if len(dataset) == 0:
        raise ValueError(f"Dataset is empty. Please check the path {root}")

    rng = np.random.default_rng(seed)
    train_ind = rng.choice(len(dataset), size=int(len(dataset)*ratio_train), replace=False)
    val_ind = np.setdiff1d(np.arange(len(dataset)), train_ind)

    train_dataset = torch.utils.data.Subset(dataset, train_ind)
    val_dataset = torch.utils.data.Subset(dataset_val, val_ind)
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError(f"Train or validation dataset is empty. Please check the path {root}")
    return train_dataset, val_dataset
