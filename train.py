"""
train.py

Stratified 5-fold cross-validation, ResNet18 transfer learning, on the
6-subtype dataset produced by prepare_data.py:
  elliptical_round, elliptical_inbetween, elliptical_cigar,
  spiral_face_on, spiral_barred, spiral_edge_on

Every image gets predicted exactly once, by a model that never trained on it,
so the reported accuracy covers the full dataset rather than one lucky split.
Class-weighted loss (inverse frequency) is used throughout to counter the
heavy imbalance toward elliptical_inbetween.

Reports:
  - Per-fold and mean±std 6-way subtype accuracy
  - Binary spiral-detection precision/recall (spiral_* classes vs. the rest)
  - Confusion matrix (saved as PNG)

Usage:
  python train.py --data_dir /workspace/data/subtypes_v2 --epochs 15
  python train.py --data_dir /workspace/data/subtypes_v2 --final   # train one deployable model on all data
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score

SPIRAL_CLASSES = {"spiral_face_on", "spiral_barred", "spiral_edge_on"}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(train: bool):
    ops = [transforms.Resize((224, 224))]
    if train:
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
        ]
    ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(ops)


def class_weights(targets: np.ndarray, num_classes: int, device: torch.device) -> torch.Tensor:
    """Inverse-frequency weights so the majority class (elliptical_inbetween etc.)
    doesn't dominate the loss. weight_c = total_samples / (num_classes * count_c)."""
    counts = np.bincount(targets, minlength=num_classes).astype(float)
    counts[counts == 0] = 1  # guard against empty classes
    weights = len(targets) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


class TransformSubset(torch.utils.data.Dataset):
    """Wraps a fixed list of (path, label) samples with a given transform.

    Used instead of re-instantiating ImageFolder per fold: re-scanning the
    directory from disk for each fold risks index misalignment whenever the
    in-memory sample list has been filtered (e.g. --max_per_class), since a
    fresh ImageFolder always re-scans the FULL unfiltered directory. Building
    every fold's Dataset from the same already-filtered `samples` list makes
    train/val indices always refer to the same underlying images regardless
    of filtering.
    """

    def __init__(self, samples, loader, transform):
        self.samples = samples
        self.loader = loader
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def build_model(num_classes: int, device: torch.device) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze the last residual block so the model can adapt mid-level features
    # to galaxy morphology, not just reuse generic ImageNet features through a
    # frozen backbone -- a single trainable linear head plateaus quickly (see
    # v1 run: train/val both flatlined ~60% from epoch 5 on, indicating an
    # underfitting capacity ceiling, not an overfitting or LR problem).
    for param in model.layer4.parameters():
        param.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # always trainable
    return model.to(device)


def build_optimizer(model: nn.Module, lr: float) -> torch.optim.Optimizer:
    """Differential learning rates: the newly-initialized head needs the full
    lr, but layer4 is pretrained and already close to a good solution, so it
    only needs gentle fine-tuning (1/10th lr) to avoid destroying those
    pretrained features."""
    head_params = list(model.fc.parameters())
    backbone_params = [p for n, p in model.named_parameters()
                        if n.startswith("layer4") and p.requires_grad]
    return torch.optim.Adam([
        {"params": head_params, "lr": lr},
        {"params": backbone_params, "lr": lr * 0.1},
    ])


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == targets).sum().item()
            n += images.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, targets_all = [], []
    for images, targets in loader:
        images = images.to(device)
        outputs = model(images)
        preds.append(outputs.argmax(1).cpu().numpy())
        targets_all.append(targets.numpy())
    return np.concatenate(preds), np.concatenate(targets_all)


def train_one_fold(train_ds, val_ds, num_classes, device, epochs, lr, batch_size, weights):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    model = build_model(num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = build_optimizer(model, lr)

    for epoch in range(epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        print(f"    epoch {epoch+1:2d}/{epochs}  train_loss={train_loss:.3f} train_acc={train_acc:.3f}"
              f"  val_loss={val_loss:.3f} val_acc={val_acc:.3f}")

    return model, val_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/workspace/data/subtypes_v2")
    parser.add_argument("--out_dir", default="/workspace/checkpoints")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--final", action="store_true",
                         help="Skip CV, train one model on all data, save checkpoint")
    parser.add_argument("--max_per_class", type=int, default=None,
                         help="Optional cap per class for a quick smoke-test run")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    full_dataset = datasets.ImageFolder(args.data_dir, transform=build_transforms(train=False))
    classes = full_dataset.classes
    num_classes = len(classes)
    spiral_idx = {i for i, c in enumerate(classes) if c in SPIRAL_CLASSES}
    print(f"Classes: {classes}")
    print(f"Total images: {len(full_dataset)}")

    targets = np.array([label for _, label in full_dataset.samples])

    if args.max_per_class:
        keep = []
        for c in range(num_classes):
            idx = np.where(targets == c)[0][: args.max_per_class]
            keep.extend(idx.tolist())
        keep = np.array(sorted(keep))
        full_dataset.samples = [full_dataset.samples[i] for i in keep]
        full_dataset.targets = [full_dataset.targets[i] for i in keep]
        targets = np.array(full_dataset.targets)
        print(f"Smoke-test subset: {len(full_dataset)} images")

    if args.final:
        print("\nTraining final deployable model on all data...")
        train_ds = full_dataset
        train_ds.transform = build_transforms(train=True)
        loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
        model = build_model(num_classes, device)
        weights = class_weights(targets, num_classes, device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = build_optimizer(model, args.lr)
        for epoch in range(args.epochs):
            loss, acc = run_epoch(model, loader, criterion, optimizer, device, train=True)
            print(f"  epoch {epoch+1:2d}/{args.epochs}  loss={loss:.3f} acc={acc:.3f}")
        ckpt_path = out_dir / "final_model.pt"
        torch.save({"model_state": model.state_dict(), "classes": classes}, ckpt_path)
        print(f"\nSaved final model to {ckpt_path}")
        return

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    all_preds = np.zeros(len(full_dataset), dtype=int)
    fold_accuracies = []

    start = time.time()
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
        print(f"\n=== Fold {fold+1}/{args.folds} "
              f"(train={len(train_idx)}, val={len(val_idx)}) ===")

        train_samples = [full_dataset.samples[i] for i in train_idx]
        train_subset = TransformSubset(train_samples, full_dataset.loader, build_transforms(train=True))
        val_subset = Subset(full_dataset, val_idx)

        fold_weights = class_weights(targets[train_idx], num_classes, device)

        model, val_loader = train_one_fold(
            train_subset, val_subset, num_classes, device, args.epochs, args.lr, args.batch_size,
            fold_weights
        )
        preds, fold_targets = predict(model, val_loader, device)
        all_preds[val_idx] = preds

        fold_acc = accuracy_score(fold_targets, preds)
        fold_accuracies.append(fold_acc)
        print(f"  Fold {fold+1} accuracy: {fold_acc:.4f}")

    elapsed = time.time() - start
    print(f"\nTotal CV time: {elapsed/60:.1f} min")

    # Every image now has an out-of-fold prediction -> full-dataset metrics
    overall_acc = accuracy_score(targets, all_preds)
    print(f"\n=== 6-way subtype accuracy (out-of-fold, full dataset) ===")
    print(f"Mean fold accuracy: {np.mean(fold_accuracies):.4f} +/- {np.std(fold_accuracies):.4f}")
    print(f"Overall (pooled)   : {overall_acc:.4f}")

    is_spiral_true = np.isin(targets, list(spiral_idx)).astype(int)
    is_spiral_pred = np.isin(all_preds, list(spiral_idx)).astype(int)
    precision = precision_score(is_spiral_true, is_spiral_pred)
    recall = recall_score(is_spiral_true, is_spiral_pred)
    print(f"\n=== Binary spiral detection (spiral_* vs rest) ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")

    cm = confusion_matrix(targets, all_preds)
    print(f"\n=== Confusion matrix (rows=true, cols=pred) ===")
    print("Classes:", classes)
    print(cm)

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(num_classes)); ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticks(range(num_classes)); ax.set_yticklabels(classes)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
        print(f"\nSaved confusion matrix plot to {out_dir / 'confusion_matrix.png'}")
    except ImportError:
        pass

    results = {
        "classes": classes,
        "fold_accuracies": fold_accuracies,
        "mean_fold_accuracy": float(np.mean(fold_accuracies)),
        "std_fold_accuracy": float(np.std(fold_accuracies)),
        "overall_accuracy": float(overall_acc),
        "spiral_precision": float(precision),
        "spiral_recall": float(recall),
        "confusion_matrix": cm.tolist(),
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
