import torch
import numpy as np


def mixup_data(x, y, alpha=1.0):

    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]

    return mixed_x, y, y[index], lam


def cutmix_data(x, y, alpha=1.0):
    """CutMix data augmentation."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    # Random box coordinates
    height, width = x.size(2), x.size(3)
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(width * cut_rat)
    cut_h = int(height * cut_rat)

    cx = np.random.randint(width)
    cy = np.random.randint(height)

    bbx1 = np.clip(cx - cut_w // 2, 0, width)
    bby1 = np.clip(cy - cut_h // 2, 0, height)
    bbx2 = np.clip(cx + cut_w // 2, 0, width)
    bby2 = np.clip(cy + cut_h // 2, 0, height)

    # Adjust lambda to match the actual area of the box
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (width * height))

    # Apply CutMix
    mixed_x = x.clone()
    mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]

    return mixed_x, y, y[index], lam


def mixup_loss(criterion, pred, y_a, y_b, lam):

    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
