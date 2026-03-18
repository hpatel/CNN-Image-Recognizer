import torch
import torch.nn as nn
import torch.optim as optim

from configs.config import *
from data.datamodule import get_dataloaders
from models.resnet import ResNet
from training.trainer import train_epoch


def main():

    train_loader, val_loader, test_loader = get_dataloaders()

    model = ResNet().to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )

    for epoch in range(EPOCHS):

        loss,acc = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            DEVICE
        )
        val_acc = evaluate(model, val_loader, DEVICE)
        print(f"Epoch {epoch+1}")
        print(f"Train Loss: {loss:.4f}")
        print(f"Train Acc: {acc:.4f}")
        print(f"Val Acc: {val_acc:.4f}")
        
        torch.save(model.state_dict(),"resnet_cifar10.pth")


if __name__ == "__main__":
    main()