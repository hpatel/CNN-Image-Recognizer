import torch
import torch.nn as nn
import torch.optim as optim

from models.model import ResNet
from data.datamodule import get_dataloaders
from training.trainer import train_epoch
from training.evaluate import evaluate
from utils.seed import set_seed

from configs.config import *

set_seed(SEED)

train_loader, val_loader, test_loader = get_dataloaders()

model = ResNet(NUM_CLASSES).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(
    model.parameters(),
    lr=LR,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=LR_MIN
)

best_val_acc = 0
patience_counter = 0

for epoch in range(EPOCHS):

    loss, acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE)

    val_acc = evaluate(model, val_loader, DEVICE)

    current_lr = optimizer.param_groups[0]['lr']

    print(f"\nEpoch {epoch+1}")
    print(f"Train Loss: {loss:.4f}")
    print(f"Train Acc: {acc:.4f}")
    print(f"Val Acc: {val_acc:.4f}")
    print(f"LR: {current_lr:.6f}")

    scheduler.step()

    if val_acc > best_val_acc + MIN_DELTA:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pth")
        print("✅ Best model saved")
    else:
        patience_counter += 1
        print(f"⏳ No improvement ({patience_counter}/{EARLY_STOPPING_PATIENCE})")

    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print("🛑 Early stopping")
        break

# ---- FINAL TEST ----
model.load_state_dict(torch.load("best_model.pth"))

test_acc = evaluate(model, test_loader, DEVICE)
print(f"\n🎯 Final Test Accuracy: {test_acc:.4f}")