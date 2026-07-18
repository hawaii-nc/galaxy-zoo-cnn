"""
prepare_data.py

Builds a 5-subtype spiral-detection dataset from the galaxy-datasets GZ2 catalog.

Input:  the pandas catalog + label_cols returned by `galaxy_datasets.gz2(...)`
        (vote COUNTS per answer, e.g. 'smooth-or-featured-gz2_smooth')
Output: <out_dir>/{spiral_face_on,spiral_barred,spiral_edge_on,elliptical,disk_no_arms}/*.jpg
        (symlinks to the original images, so this is fast and doesn't duplicate 3GB of data)

GZ2 decision tree logic used here:
  smooth-or-featured: smooth vs featured-or-disk vs artifact
    -> smooth (high frac)                                => elliptical
    -> featured-or-disk (high frac):
         disk-edge-on: yes (high frac)                    => spiral_edge_on
         disk-edge-on: no (high frac):
              has-spiral-arms: no (high frac)              => disk_no_arms
              has-spiral-arms: yes (high frac):
                   bar: yes (high frac)                    => spiral_barred
                   bar: no (high frac)                     => spiral_face_on

Each branch requires BOTH the parent answer's fraction AND the child answer's
fraction to clear --threshold, and a minimum vote count (--min_votes) so a
galaxy with 3 total votes can't produce a "confident" label by chance.
"""

import argparse
import os
import shutil
from pathlib import Path

import pandas as pd
from galaxy_datasets import gz2


def fraction(catalog: pd.DataFrame, question_cols: list[str], answer_col: str) -> pd.Series:
    """Vote fraction for one answer = count(answer) / sum(count(all answers to that question))."""
    total = catalog[question_cols].sum(axis=1).replace(0, pd.NA)
    return catalog[answer_col] / total


def build_labels(catalog: pd.DataFrame, threshold: float, min_votes: int) -> pd.Series:
    smooth_cols = [
        "smooth-or-featured-gz2_smooth",
        "smooth-or-featured-gz2_featured-or-disk",
        "smooth-or-featured-gz2_artifact",
    ]
    edge_cols = ["disk-edge-on-gz2_yes", "disk-edge-on-gz2_no"]
    spiral_cols = ["has-spiral-arms-gz2_yes", "has-spiral-arms-gz2_no"]
    bar_cols = ["bar-gz2_yes", "bar-gz2_no"]

    f_smooth = fraction(catalog, smooth_cols, "smooth-or-featured-gz2_smooth")
    f_featured = fraction(catalog, smooth_cols, "smooth-or-featured-gz2_featured-or-disk")
    f_edge_yes = fraction(catalog, edge_cols, "disk-edge-on-gz2_yes")
    f_edge_no = fraction(catalog, edge_cols, "disk-edge-on-gz2_no")
    f_spiral_yes = fraction(catalog, spiral_cols, "has-spiral-arms-gz2_yes")
    f_spiral_no = fraction(catalog, spiral_cols, "has-spiral-arms-gz2_no")
    f_bar_yes = fraction(catalog, bar_cols, "bar-gz2_yes")
    f_bar_no = fraction(catalog, bar_cols, "bar-gz2_no")

    total_votes = catalog[smooth_cols].sum(axis=1)
    enough_votes = total_votes >= min_votes

    label = pd.Series(index=catalog.index, dtype=object)

    elliptical = enough_votes & (f_smooth >= threshold)
    label[elliptical] = "elliptical"

    featured = enough_votes & (f_featured >= threshold)

    edge_on = featured & (f_edge_yes >= threshold)
    label[edge_on] = "spiral_edge_on"

    not_edge_on = featured & (f_edge_no >= threshold)

    no_arms = not_edge_on & (f_spiral_no >= threshold)
    label[no_arms] = "disk_no_arms"

    has_arms = not_edge_on & (f_spiral_yes >= threshold)

    barred = has_arms & (f_bar_yes >= threshold)
    label[barred] = "spiral_barred"

    face_on_unbarred = has_arms & (f_bar_no >= threshold)
    label[face_on_unbarred] = "spiral_face_on"

    return label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/workspace/data/gz2",
                         help="Root passed to galaxy_datasets.gz2(root=...)")
    parser.add_argument("--out_dir", default="/workspace/data/subtypes",
                         help="Where to write the 5 label subfolders")
    parser.add_argument("--threshold", type=float, default=0.8,
                         help="Minimum vote fraction to accept a label at each tree node")
    parser.add_argument("--min_votes", type=int, default=10,
                         help="Minimum total votes on the smooth-or-featured question")
    parser.add_argument("--copy", action="store_true",
                         help="Copy images instead of symlinking (slower, uses more disk)")
    args = parser.parse_args()

    print(f"Loading GZ2 catalog from {args.data_root} ...")
    catalog, label_cols = gz2(root=args.data_root, train=True, download=False)
    print(f"Loaded {len(catalog)} galaxies, {len(label_cols)} label columns")

    labels = build_labels(catalog, args.threshold, args.min_votes)
    catalog = catalog.assign(subtype=labels)
    labeled = catalog.dropna(subset=["subtype"])

    print(f"\nLabel counts at threshold={args.threshold}, min_votes={args.min_votes}:")
    counts = labeled["subtype"].value_counts()
    print(counts)
    print(f"\nTotal labeled: {len(labeled)} / {len(catalog)} "
          f"({100 * len(labeled) / len(catalog):.1f}%)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for subtype in counts.index:
        (out_dir / subtype).mkdir(exist_ok=True)

    print(f"\nWriting to {out_dir} ({'copy' if args.copy else 'symlink'}) ...")
    n_written = 0
    n_missing = 0
    for _, row in labeled.iterrows():
        src = Path(row["file_loc"])
        if not src.exists():
            n_missing += 1
            continue
        dst = out_dir / row["subtype"] / src.name
        if dst.exists() or dst.is_symlink():
            continue
        if args.copy:
            shutil.copy2(src, dst)
        else:
            os.symlink(src.resolve(), dst)
        n_written += 1

    print(f"Wrote {n_written} images ({n_missing} source files missing on disk)")
    print("\nDone. Class distribution:")
    for subtype in sorted(counts.index):
        n = len(list((out_dir / subtype).iterdir()))
        print(f"  {subtype:16s} {n}")


if __name__ == "__main__":
    main()
