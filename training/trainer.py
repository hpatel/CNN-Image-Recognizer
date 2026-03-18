import torch
from tqdm import tqdm
from utils.metrics import accuracy


def train_epoch(model,loader,optimizer,criterion,device):

    model.train()

    total_loss = 0
    total_acc = 0

    loop = tqdm(loader)

    for images,labels in loop:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = accuracy(outputs,labels)

        total_loss += loss.item()
        total_acc += acc

    return total_loss/len(loader), total_acc/len(loader)