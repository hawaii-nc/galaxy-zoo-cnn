"""
evaluate_decals.py

Generalization test: takes the model trained on GZ2 (SDSS imaging) and
evaluates it on GZ DECaLS GZD-5 (deeper, different-telescope imaging), with
labels built independently from DECaLS's own volunteer vote fractions using
the same hierarchical/confidence-threshold logic as prepare_data.py.

This is a genuine out-of-distribution test: different survey, different
imaging pipeline, different (though overlapping) volunteer population --
not just another held-out split of the same GZ2 images. If accuracy holds up
here, the model learned real morphology, not SDSS-specific image quirks.

DECaLS schema differences from GZ2, handled below:
  - bar-dr5 is three-way (strong/weak/no) vs GZ2's binary (yes/no).
    strong or weak together are treated as "has bar" to match the
    spiral_barred definition used during training.
  - how-rounded-dr5_cigar-shaped (vs GZ2's how-rounded-gz2_cigar) -- same
    concept, different column suffix.

Usage:
  python evaluate_decals.py --checkpoint /workspace/checkpoints/final_model.pt \
      --data_root /workspace/data/gz_decals
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from galaxy_datasets import gz_decals_5

SPIRAL_CLASSES = {"spiral_face_on", "spiral_barred", "spiral_edge_on"}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def fraction(catalog: pd.DataFrame, question_cols: list[str], answer_col: str) -> pd.Series:
    total = catalog[question_cols].sum(axis=1).replace(0, pd.NA)
    return catalog[answer_col] / total


def build_labels(catalog: pd.DataFrame, threshold: float, min_votes: int) -> pd.Series:
    smooth_cols = [
        "smooth-or-featured-dr5_smooth",
        "smooth-or-featured-dr5_featured-or-disk",
        "smooth-or-featured-dr5_artifact",
    ]
    edge_cols = ["disk-edge-on-dr5_yes", "disk-edge-on-dr5_no"]
    spiral_cols = ["has-spiral-arms-dr5_yes", "has-spiral-arms-dr5_no"]
    bar_cols = ["bar-dr5_strong", "bar-dr5_weak", "bar-dr5_no"]
    round_cols = ["how-rounded-dr5_round", "how-rounded-dr5_in-between", "how-rounded-dr5_cigar-shaped"]

    f_smooth = fraction(catalog, smooth_cols, "smooth-or-featured-dr5_smooth")
    f_featured = fraction(catalog, smooth_cols, "smooth-or-featured-dr5_featured-or-disk")
    f_edge_yes = fraction(catalog, edge_cols, "disk-edge-on-dr5_yes")
    f_edge_no = fraction(catalog, edge_cols, "disk-edge-on-dr5_no")
    f_spiral_yes = fraction(catalog, spiral_cols, "has-spiral-arms-dr5_yes")
    f_spiral_no = fraction(catalog, spiral_cols, "has-spiral-arms-dr5_no")

    # bar is three-way in DECaLS (strong/weak/no) -- combine strong+weak as "has bar"
    # to match the binary yes/no bar definition spiral_barred was trained on.
    bar_total = catalog[bar_cols].sum(axis=1).replace(0, pd.NA)
    f_bar_yes = (catalog["bar-dr5_strong"] + catalog["bar-dr5_weak"]) / bar_total
    f_bar_no = catalog["bar-dr5_no"] / bar_total

    f_round = fraction(catalog, round_cols, "how-rounded-dr5_round")
    f_inbetween = fraction(catalog, round_cols, "how-rounded-dr5_in-between")
    f_cigar = fraction(catalog, round_cols, "how-rounded-dr5_cigar-shaped")

    total_votes = catalog[smooth_cols].sum(axis=1)
    enough_votes = total_votes >= min_votes

    label = pd.Series(index=catalog.index, dtype=object)

    smooth = enough_votes & (f_smooth >= threshold)
    label[smooth & (f_round >= threshold)] = "elliptical_round"
    label[smooth & (f_inbetween >= threshold)] = "elliptical_inbetween"
    label[smooth & (f_cigar >= threshold)] = "elliptical_cigar"

    featured = enough_votes & (f_featured >= threshold)
    label[featured & (f_edge_yes >= threshold)] = "spiral_edge_on"

    not_edge_on = featured & (f_edge_no >= threshold)
    has_arms = not_edge_on & (f_spiral_yes >= threshold)
    label[has_arms & (f_bar_yes >= threshold)] = "spiral_barred"
    label[has_arms & (f_bar_no >= threshold)] = "spiral_face_on"

    return label


class EvalDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples  # list of (path, label_idx)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image
        path, target = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), target


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)  # weights loaded from checkpoint, not ImageNet
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/workspace/checkpoints/final_model.pt")
    parser.add_argument("--data_root", default="/workspace/data/gz_decals")
    parser.add_argument("--out_dir", default="/workspace/checkpoints")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--min_votes", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading checkpoint...")
    ckpt = torch.load(args.checkpoint, map_location=device)
    classes = ckpt["classes"]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"Model classes (from GZ2 training): {classes}")

    model = build_model(len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    print(f"\nLoading DECaLS catalog from {args.data_root} ...")
    catalog, label_cols = gz_decals_5(root=args.data_root, train=True, download=False)
    print(f"Loaded {len(catalog)} DECaLS galaxies")

    labels = build_labels(catalog, args.threshold, args.min_votes)
    catalog = catalog.assign(subtype=labels)
    labeled = catalog.dropna(subset=["subtype"])
    print(f"\nLabel counts at threshold={args.threshold}, min_votes={args.min_votes}:")
    print(labeled["subtype"].value_counts())
    print(f"\nTotal labeled: {len(labeled)} / {len(catalog)} "
          f"({100 * len(labeled) / len(catalog):.1f}%)")

    samples = []
    n_missing = 0
    for _, row in labeled.iterrows():
        path = Path(row["file_loc"])
        if not path.exists():
            n_missing += 1
            continue
        samples.append((str(path), class_to_idx[row["subtype"]]))
    if n_missing:
        print(f"({n_missing} images missing on disk, skipped)")
    print(f"Evaluating on {len(samples)} DECaLS images never seen during GZ2 training")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    dataset = EvalDataset(samples, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            outputs = model(images)
            all_preds.append(outputs.argmax(1).cpu().numpy())
            all_targets.append(targets.numpy())
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    acc = accuracy_score(all_targets, all_preds)
    print(f"\n=== DECaLS generalization accuracy (6-way) ===")
    print(f"Accuracy: {acc:.4f}")

    spiral_idx = {i for i, c in enumerate(classes) if c in SPIRAL_CLASSES}
    is_spiral_true = np.isin(all_targets, list(spiral_idx)).astype(int)
    is_spiral_pred = np.isin(all_preds, list(spiral_idx)).astype(int)
    precision = precision_score(is_spiral_true, is_spiral_pred)
    recall = recall_score(is_spiral_true, is_spiral_pred)
    print(f"\n=== Binary spiral detection on DECaLS ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")

    cm = confusion_matrix(all_targets, all_preds, labels=list(range(len(classes))))
    print(f"\n=== Confusion matrix (rows=true, cols=pred) ===")
    print("Classes:", classes)
    print(cm)

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm, cmap="Oranges")
        ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title("GZ2-trained model evaluated on DECaLS (unseen survey)")
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.colorbar(im)
        fig.tight_layout()
        out_path = Path(args.out_dir) / "decals_confusion_matrix.png"
        fig.savefig(out_path, dpi=150)
        print(f"\nSaved confusion matrix plot to {out_path}")
    except ImportError:
        pass

    import json
    results = {
        "n_evaluated": len(samples),
        "accuracy": float(acc),
        "spiral_precision": float(precision),
        "spiral_recall": float(recall),
        "confusion_matrix": cm.tolist(),
        "classes": classes,
    }
    out_path = Path(args.out_dir) / "decals_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
