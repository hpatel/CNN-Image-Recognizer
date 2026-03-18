import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

from configs.config import BATCH_SIZE, VAL_SPLIT, TEST_SPLIT


def get_dataloaders():

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914,0.4822,0.4465),
            (0.2023,0.1994,0.2010)
        )
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914,0.4822,0.4465),
            (0.2023,0.1994,0.2010)
        )
    ])

    # ===== FULL TRAIN DATASET =====
    full_train = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform_train
    )

    # ===== SPLIT INTO TRAIN + VAL =====
    train_size = int((1 - VAL_SPLIT) * len(full_train))
    val_size = len(full_train) - train_size

    train_dataset, val_dataset = random_split(
        full_train,
        [train_size, val_size]
    )

    # ===== TEST DATASET =====
    full_test = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform_test
    )

    # ===== LOADERS =====
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )

    return train_loader, val_loader, test_loader