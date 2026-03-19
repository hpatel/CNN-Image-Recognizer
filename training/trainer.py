import torch
from utils.mixup import mixup_data, cutmix_data, mixup_loss


def train_epoch(model, loader, optimizer, criterion, device):

    model.train()

    total_loss = 0
    total_acc = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        # Randomly choose between MixUp and CutMix
        if torch.rand(1).item() > 0.5:
            images, y_a, y_b, lam = mixup_data(images, labels)
        else:
            images, y_a, y_b, lam = cutmix_data(images, labels)

        outputs = model(images)
        loss = mixup_loss(criterion, outputs, y_a, y_b, lam)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        acc = (preds == labels).float().mean().item()

        total_loss += loss.item()
        total_acc += acc

    return total_loss / len(loader), total_acc / len(loader)