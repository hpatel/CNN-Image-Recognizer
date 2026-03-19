# CNN Image Recognizer

A simple PyTorch project that trains a ResNet-style CNN on the CIFAR-10 dataset.

This repository includes a minimal training loop, data loading pipeline, and a small ResNet implementation.

---

## ✅ Features

- ✅ Trains a ResNet-like CNN from scratch on CIFAR-10
- ✅ Uses PyTorch and torchvision
- ✅ Includes configurable hyperparameters (learning rate, batch size, epochs, etc.)
- ✅ Saves model weights to `best_model.pth`

---

## 📦 Requirements

- Python 3.8+
- PyTorch (compatible with your CUDA version, or CPU-only)
- torchvision

Install dependencies via:

```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start

1) From the project root:

```bash
python main.py
```

This will:
- Download/prepare CIFAR-10 data into `./data/`
- Train a small ResNet model for the configured number of epochs
- Save the best model weights to `best_model.pth`

---

## 🧠 Inference (Prediction)

Once training completes, run inference on a new image with:

```bash
python predict.py --image path/to/image.png --weights best_model.pth
```

The script will print the predicted CIFAR-10 class label.

---

## ⚙️ Configuration

Adjust training settings in `configs/config.py`:

- `BATCH_SIZE`: training batch size
- `LR`: learning rate
- `EPOCHS`: number of training epochs
- `DEVICE`: `"cuda"` or `"cpu"`
- `WEIGHT_DECAY`, `MOMENTUM`
- `VAL_SPLIT`: validation split ratio

---

## 🧠 Project Structure

```
CNN-Image-Recognizer/
├─ configs/
│  └─ config.py          # hyperparameters + device settings
├─ data/
│  └─ datamodule.py     # CIFAR-10 dataloaders (train/val/test)
├─ models/
│  └─ model.py          # ResNet-like CNN definition
├─ training/
│  ├─ trainer.py        # training loop (mixup) + optimizer/scheduler
│  └─ evaluate.py       # evaluation helper
├─ utils/
│  └─ seed.py           # deterministic seeding utility
├─ predict.py           # inference helper script
└─ main.py              # entry point for training
```

---

## 📝 Notes

- The current implementation uses CIFAR-10 (10 classes).
- The model checkpoint is overwritten (when validation improves) at `best_model.pth`.
- For evaluation or inference, load the saved weights into `models.model.ResNet()` and call `model.eval()`.

---

## 🧪 Extending This Project

Ideas to expand:

- Add CLI arguments for training parameters and checkpoint paths
- Improve `predict.py` (e.g., add batched inference / top-k predictions)
- Add better logging (TensorBoard / Weights & Biases)
- Add learning rate scheduling and early stopping

---

Happy training! 👋
