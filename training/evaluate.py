import torch


def evaluate(model, loader, device):

    model.eval()
    total_acc = 0

    with torch.no_grad():
        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            acc = (preds == labels).float().mean().item()
            total_acc += acc

    return total_acc / len(loader)