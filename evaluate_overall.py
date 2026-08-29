"""evaluate_overall.py -- End-to-end evaluation of the WHOLE fused system
(YOLO species detection -> colour KNN -> defect module -> _fuse_quality),
NOT just the colour classifier on its own.

This does not retrain or modify anything. It runs overall.run_overall_pipeline()
on real labelled images from ImageProcessing_ASS/dataset/train and compares the
fused final_quality against the folder's ground-truth label.

Usage:
    python evaluate_overall.py --dataset-dir "D:\\Image Processing\\ImageProcessing_ASS\\dataset\\train" --max-per-class 20 --log-csv
"""
import argparse
import glob
import json
import os
import random
import sys
import csv
from collections import defaultdict

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import overall  # noqa: E402

# folder name (relative to dataset-dir) -> (fruit, ground-truth quality)
CLASS_FOLDERS = {
    "freshapples": ("Apple", "Fresh"),
    "rottenapples": ("Apple", "Rotten"),
    "unripe apple": ("Apple", "Unripe"),
    "freshbanana": ("Banana", "Fresh"),
    "rottenbanana": ("Banana", "Rotten"),
    "unripe banana": ("Banana", "Unripe"),
    "ripemango": ("Mango", "Fresh"),
    "rottonmango": ("Mango", "Rotten"),
    "unripemango": ("Mango", "Unripe"),
    "freshoranges": ("Orange", "Fresh"),
    "rottenoranges": ("Orange", "Rotten"),
    "unripe orange": ("Orange", "Unripe"),
    "FreshStrawberry": ("Strawberry", "Fresh"),
    "RottenStrawberry": ("Strawberry", "Rotten"),
    "unripe strawberry": ("Strawberry", "Unripe"),
}

QUALITY_CLASSES = ["Fresh", "Unripe", "Rotten"]
EXTRA_PRED_LABELS = ["Uncertain", "NoDetection"]

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def iter_dataset(dataset_dir, max_per_class=None, seed=42):
    rng = random.Random(seed)
    for folder, (fruit, quality) in CLASS_FOLDERS.items():
        folder_path = os.path.join(dataset_dir, folder)
        if not os.path.isdir(folder_path):
            print(f"WARNING: missing folder {folder_path}", file=sys.stderr)
            continue
        paths = [
            p for p in glob.glob(os.path.join(folder_path, "*"))
            if p.lower().endswith(IMG_EXTS)
        ]
        rng.shuffle(paths)
        if max_per_class is not None:
            paths = paths[:max_per_class]
        for p in paths:
            yield p, fruit, quality


def pick_best_match(results, expected_fruit):
    """Among this image's FruitResults, return the one whose detected
    species matches the expected fruit (highest species_confidence if
    several). Returns None if nothing matches -- caller treats this as a
    species mismatch (a real fruit was found, just not classified as the
    expected species)."""
    candidates = [r for r in results if r.species == expected_fruit]
    if not candidates:
        return None
    candidates.sort(key=lambda r: r.species_confidence or 0.0, reverse=True)
    return candidates[0]


def compute_metrics(y_true, y_pred):
    labels = QUALITY_CLASSES + [lbl for lbl in EXTRA_PRED_LABELS if lbl in y_pred]
    return {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred) if y_true else 0.0,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred) if y_true else 0.0,
        "macro_f1": f1_score(y_true, y_pred, labels=QUALITY_CLASSES, average="macro", zero_division=0) if y_true else 0.0,
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist() if y_true else [],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", required=True, help=r'Path to ImageProcessing_ASS\dataset\train')
    ap.add_argument("--max-per-class", type=int, default=40, help="Cap images per class-folder (default 40)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="evaluation_results")
    ap.add_argument("--log-csv", action="store_true", help="Write a per-image diagnostic CSV log")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    y_true, y_pred = [], []
    per_fruit = defaultdict(lambda: {"y_true": [], "y_pred": []})
    n_requested = 0
    no_detection = 0
    species_mismatch = 0
    log_rows = []

    for path, expected_fruit, expected_quality in iter_dataset(args.dataset_dir, args.max_per_class, args.seed):
        n_requested += 1
        image_bgr = cv2.imread(path)
        if image_bgr is None:
            print(f"WARNING: could not read {path}", file=sys.stderr)
            continue

        try:
            results, _resized = overall.run_overall_pipeline(image_bgr)
        except Exception as exc:
            print(f"WARNING: pipeline error on {path}: {exc}", file=sys.stderr)
            results = []

        if not results:
            no_detection += 1
            predicted_quality = "NoDetection"
            predicted_species = None
            colour_quality = colour_confidence = None
            defect_ripeness = defect_percentage = None
            stem_detected = None
            quality_note = "no fruit detected by pipeline"
        else:
            match = pick_best_match(results, expected_fruit)
            if match is None:
                species_mismatch += 1
                predicted_quality = "NoDetection"
                predicted_species = results[0].species
                colour_quality = colour_confidence = None
                defect_ripeness = defect_percentage = None
                stem_detected = None
                quality_note = f"species mismatch (got {results[0].species}, expected {expected_fruit})"
            else:
                predicted_quality = match.final_quality or "Uncertain"
                predicted_species = match.species
                colour_quality = match.colour_quality
                colour_confidence = match.colour_confidence
                defect_ripeness = match.defect_ripeness
                defect_percentage = match.defect_percentage
                stem_detected = match.stem_detected
                quality_note = match.quality_note

        y_true.append(expected_quality)
        y_pred.append(predicted_quality)
        per_fruit[expected_fruit]["y_true"].append(expected_quality)
        per_fruit[expected_fruit]["y_pred"].append(predicted_quality)

        if args.log_csv:
            log_rows.append({
                "path": path,
                "expected_fruit": expected_fruit,
                "expected_quality": expected_quality,
                "predicted_species": predicted_species,
                "predicted_quality": predicted_quality,
                "correct": predicted_quality == expected_quality,
                "colour_quality": colour_quality,
                "colour_confidence": colour_confidence,
                "defect_ripeness": defect_ripeness,
                "defect_percentage": defect_percentage,
                "stem_detected": stem_detected,
                "quality_note": quality_note,
            })

        print(f"[{n_requested}] {expected_fruit}/{expected_quality} -> {predicted_species}/{predicted_quality}"
              f"{'  OK' if predicted_quality == expected_quality else '  WRONG'}")

    summary = {
        "n_requested": n_requested,
        "no_detection": no_detection,
        "species_mismatch": species_mismatch,
        "overall": compute_metrics(y_true, y_pred),
        "per_fruit": {
            fruit: compute_metrics(d["y_true"], d["y_pred"])
            for fruit, d in per_fruit.items()
        },
    }

    with open(os.path.join(args.out_dir, "overall_evaluation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nWrote", os.path.join(args.out_dir, "overall_evaluation_summary.json"))

    if args.log_csv:
        csv_path = os.path.join(args.out_dir, "per_image_log.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()) if log_rows else [])
            writer.writeheader()
            writer.writerows(log_rows)
        print("Wrote", csv_path)

    # --- bar chart: accuracy / balanced accuracy / macro F1, overall + per fruit ---
    fruits = sorted(per_fruit.keys())
    groups = ["Overall"] + fruits
    metrics_keys = ["accuracy", "balanced_accuracy", "macro_f1"]
    values = {mk: [summary["overall"][mk] if g == "Overall" else summary["per_fruit"][g][mk] for g in groups] for mk in metrics_keys}

    # Same 3-shade blue palette + % scale + value labels as the reference
    # chart you sent (dark = Accuracy, medium = Balanced accuracy, light =
    # Macro F1).
    BAR_COLOURS = ["#1F3864", "#2E75B6", "#9DC3E6"]

    x = np.arange(len(groups))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, mk in enumerate(metrics_keys):
        pct_values = [v * 100.0 for v in values[mk]]
        bars = ax.bar(
            x + (i - 1) * width, pct_values, width,
            label=mk.replace("_", " ").title(), color=BAR_COLOURS[i],
            edgecolor="white", linewidth=0.5,
        )
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Score (%)")
    ax.set_title("Fruit Quality Assessment -- Bar Chart", fontweight="bold")
    ax.grid(axis="y", color="#E0E0E0", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    # Legend kept above the plot area (not overlapping the bars) --
    # bbox_inches="tight" on savefig makes sure it never gets clipped.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3, frameon=False)
    # Rounded border framing the whole figure, matching the reference chart.
    frame = FancyBboxPatch(
        (0.01, 0.01), 0.98, 0.98, transform=fig.transFigure,
        boxstyle="round,pad=0,rounding_size=0.02", linewidth=1.0,
        edgecolor="#999999", facecolor="none", clip_on=False,
    )
    fig.patches.append(frame)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "fig_overall_performance_bar.png"), dpi=150, bbox_inches="tight")
    print("Wrote", os.path.join(args.out_dir, "fig_overall_performance_bar.png"))

    # --- confusion matrix (overall) ---
    labels = summary["overall"]["labels"]
    cm = np.array(summary["overall"]["confusion_matrix"])
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    im = ax2.imshow(cm, cmap="Blues")
    ax2.set_xticks(range(len(labels)))
    ax2.set_yticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=45, ha="right")
    ax2.set_yticklabels(labels)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    ax2.set_title("Fruit Quality Assessment -- Confusion Matrix")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax2.text(j, i, str(cm[i, j]), ha="center", va="center",
                      color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig2.colorbar(im, ax=ax2)
    fig2.tight_layout()
    fig2.savefig(os.path.join(args.out_dir, "fig_overall_confusion_matrix.png"), dpi=150)
    print("Wrote", os.path.join(args.out_dir, "fig_overall_confusion_matrix.png"))

    print(f"\nn_requested={n_requested}  no_detection={no_detection}  species_mismatch={species_mismatch}")
    print(f"overall accuracy={summary['overall']['accuracy']:.4f}  "
          f"balanced_accuracy={summary['overall']['balanced_accuracy']:.4f}  "
          f"macro_f1={summary['overall']['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
