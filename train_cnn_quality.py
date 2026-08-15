"""
train_cnn_quality.py
==============================================
CNN-based alternative to the patch-histogram + majority-vote quality
classifier (Fresh / Unripe / Rotten), trained per fruit type.

WHY THIS EXISTS
-------------------------------------------------------------
Every feature-engineering idea tried this session on the existing
patch-based SVM (more/fewer histogram bins, more texture bins,
dropping the V channel, a "how far does this patch's hue stray from
the fruit's own average hue" feature) landed at the same ~73% held-out
accuracy for Apple Fresh/Rotten/Unripe — none moved the needle. The
common thread: that architecture classifies each 32x32 patch on its
own, then takes a majority vote across ~15-20 patches per fruit. When
rot is a SMALL localized spot on an otherwise normal-looking apple,
most patches genuinely still look Fresh, so majority vote reports
Fresh even for a rotten apple — confirmed directly: adding a synthetic
dark spot covering up to 25% of a real Fresh apple's surface never
flipped the prediction to Rotten. No amount of per-patch feature
tuning fixes that; it's a structural property of "vote per small tile,
then average," not a feature quality problem.

A CNN looks at the WHOLE fruit image at once (no tiling, no voting) —
it can learn "this is mostly normal skin EXCEPT for one patch that
looks wrong" directly, the same way a person would look at the whole
apple rather than judging it tile-by-tile. This script trains one such
CNN per fruit type (mirroring your existing quality_models[fruit_type]
structure), using transfer learning from a small pretrained backbone
(MobileNetV2) since your real, non-augmented training data is limited
(seen directly this session: only 141-327 truly unique base photos per
class after stripping out rotated/flipped/noise-augmented duplicates)
— transfer learning is specifically suited to getting reasonable
results from that little labeled data, rather than training a CNN from
scratch (which typically wants thousands of unique images per class).

SETUP (run on your own machine — NOT verified in this sandbox; see the
note at the bottom of this docstring for why)
-------------------------------------------------------------
    pip install torch torchvision

USAGE
-------------------------------------------------------------
    python train_cnn_quality.py --dataset dataset/train --out cnn_quality_models

This trains one small model per fruit type found in CLASS_FOLDERS
(Apple / Banana / Orange by default — edit CLASS_FOLDERS below if your
folder names differ, same convention as segmentation.py) and
saves each as cnn_quality_models/<FruitType>.pt.

To classify a single already-cropped/segmented fruit image with a
trained model:

    from train_cnn_quality import load_cnn_model, predict_quality
    model, classes = load_cnn_model("cnn_quality_models/Apple.pt")
    label, confidence = predict_quality(model, classes, "some_apple_crop.jpg")

-------------------------------------------------------------
NOTE ON THIS SCRIPT'S TESTING STATUS: PyTorch could not be installed
in this session's sandbox — even installing from an already-downloaded
~500MB wheel (skipping the network entirely) repeatedly timed out due
to slow disk I/O in this particular environment, the same issue hit
when setting up the YOLO integration earlier. This script is written
and ready to run, but has not been executed end-to-end here. Run it on
your own machine and share the printed accuracy numbers + any errors —
happy to debug from real output rather than guessing.
"""

import argparse
import glob
import os
import random
import re

# Mirrors segmentation.CLASS_FOLDERS — edit if your dataset
# folder names differ. Reusing the same convention keeps this script
# drop-in compatible with the dataset you already have.
CLASS_FOLDERS = {
    "freshapples":   ("Apple", "Fresh"),
    "rottenapples":  ("Apple", "Rotten"),
    "unripe apple":  ("Apple", "Unripe"),
    "freshbanana":   ("Banana", "Fresh"),
    "rottenbanana":  ("Banana", "Rotten"),
    "unripe banana": ("Banana", "Unripe"),
    "freshoranges":  ("Orange", "Fresh"),
    "rottenoranges": ("Orange", "Rotten"),
    "unripe orange": ("Orange", "Unripe"),
}

IMAGE_SIZE = 160          # CNN input resolution — MobileNetV2's native
                           # size is 224, but 160 trains noticeably
                           # faster on CPU and transfer learning tolerates
                           # the downscale fine for this task.
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.2


_AUG_PREFIX_RE = re.compile(
    r"^(rotated_by_\d+_|translation_|saltandpepper_|salt_and_pepper_|vertical_flip_|horizontal_flip_|aug_)+",
    re.IGNORECASE,
)


def _base_image_key(path):
    """
    Strips known augmentation-prefix patterns (rotated_by_15_,
    translation_, saltandpepper_, vertical_flip_, aug_, ...) from a
    filename to recover an identifier for the ORIGINAL source photo.
    Multiple augmented copies of the same photo collapse to the same
    key, so a grouped train/val split (see _group_split) can keep every
    copy of one photo on the SAME side of the split.

    Why this matters: this dataset's file counts are mostly augmented
    duplicates of a much smaller set of real photos (confirmed earlier
    this session — e.g. only ~141-327 truly unique photos per class
    despite 1000-2300+ files). Splitting by FILE instead of by ORIGINAL
    PHOTO lets a rotated/flipped/noised copy of a training image leak
    into validation — the model has effectively already seen a
    near-duplicate of that "unseen" validation image during training.
    That's exactly what produced the first run's suspicious 99%+
    validation accuracy: it wasn't measuring real generalization.
    """
    name = os.path.basename(path)
    stem, _ext = os.path.splitext(name)
    prev = None
    while prev != stem:
        prev = stem
        stem = _AUG_PREFIX_RE.sub("", stem)
    return stem.lower()


def _group_split(dataset, val_frac, seed=42):
    """
    Splits `dataset.samples` into (train_indices, val_indices) grouped
    by base-image key so every augmented copy of one original photo
    ends up on the same side of the split. Returns two lists of
    integer indices into `dataset.samples`.
    """
    groups = {}
    for idx, (_path, _label, key) in enumerate(dataset.samples):
        groups.setdefault(key, []).append(idx)

    group_keys = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_keys)

    target_val = max(1, int(len(dataset) * val_frac))
    val_indices = []
    train_indices = []
    for key in group_keys:
        idxs = groups[key]
        if len(val_indices) < target_val:
            val_indices.extend(idxs)
        else:
            train_indices.extend(idxs)

    print(f"[INFO] Grouped {len(dataset)} files into {len(group_keys)} original-photo "
          f"families -> {len(train_indices)} train / {len(val_indices)} val samples "
          f"(no family split across both sides)")
    return train_indices, val_indices


def _build_model(num_classes):
    """
    MobileNetV2 pretrained on ImageNet, with its classifier head
    replaced for our 2-3 quality classes and everything BEFORE the
    last few layers frozen. Freezing the early layers is the standard
    transfer-learning move when labeled data is limited (as it
    genuinely is here) — those layers already encode generic edge/
    texture/color features useful for almost any image task; only the
    later, more task-specific layers need retraining on your fruit
    photos.
    """
    import torch.nn as nn
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    for param in model.features[:-4].parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def _build_transforms(train: bool):
    from torchvision import transforms

    if train:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class FruitQualityDataset:
    """Loads (image_path, quality_label, base_image_key) triples for ONE
    fruit type from the dataset folder structure, matching CLASS_FOLDERS."""

    def __init__(self, dataset_dir, fruit_type, transform=None):
        # NOTE: deliberately NOT storing the PIL Image module (or any
        # module object) as an instance attribute. On Windows,
        # DataLoader(num_workers>0) pickles the whole Dataset object to
        # send it to worker processes (spawn start method) — modules
        # aren't picklable, which is exactly what caused
        # "TypeError: cannot pickle 'module' object". Importing PIL
        # fresh inside __getitem__ instead avoids that entirely.
        self.transform = transform
        self.samples = []
        self.classes = sorted({
            quality for folder, (ftype, quality) in CLASS_FOLDERS.items()
            if ftype == fruit_type
        })
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for folder, (ftype, quality) in CLASS_FOLDERS.items():
            if ftype != fruit_type:
                continue
            class_dir = os.path.join(dataset_dir, folder)
            if not os.path.isdir(class_dir):
                print(f"[WARN] Folder not found, skipping: {class_dir}")
                continue
            paths = sorted(
                glob.glob(os.path.join(class_dir, "*.jpg"))
                + glob.glob(os.path.join(class_dir, "*.jpeg"))
                + glob.glob(os.path.join(class_dir, "*.png"))
            )
            for p in paths:
                self.samples.append((p, self.class_to_idx[quality], _base_image_key(p)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image
        path, label, _key = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def train_fruit_type_model(dataset_dir, fruit_type, epochs=EPOCHS, batch_size=BATCH_SIZE):
    import torch
    from torch.utils.data import DataLoader, Subset
    import torch.nn as nn
    import torch.optim as optim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training {fruit_type} quality model on device: {device}")

    full_dataset = FruitQualityDataset(dataset_dir, fruit_type, transform=_build_transforms(train=True))
    if len(full_dataset) == 0:
        print(f"[WARN] No images found for {fruit_type}, skipping.")
        return None, None

    classes = full_dataset.classes
    print(f"[INFO] {fruit_type}: {len(full_dataset)} images, classes={classes}")

    # Grouped by original-photo identity (see _base_image_key /
    # _group_split docstrings) so augmented copies of one training
    # photo can't leak into validation — a plain random file-level
    # split let that happen on the first run of this script and
    # produced a fake 99%+ validation accuracy.
    train_indices, val_indices = _group_split(full_dataset, VAL_SPLIT, seed=42)

    # Validation set should NOT get the training-time augmentation
    # (random flips/rotation/jitter) — build it from a separately
    # instantiated dataset with the plain eval transform, using the
    # SAME file ordering (paths are sorted in __init__) so indices
    # line up with train_indices/val_indices above.
    eval_dataset = FruitQualityDataset(dataset_dir, fruit_type, transform=_build_transforms(train=False))
    train_ds = Subset(full_dataset, train_indices)
    val_ds = Subset(eval_dataset, val_indices)

    # num_workers=0 (load in the main process) rather than >0: on
    # Windows, DataLoader workers are separate processes (spawn start
    # method, no fork), which adds real pickling fragility for not
    # much speed gain on a dataset this size — simplicity over a small
    # possible speedup here.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = _build_model(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()

        model.eval()
        val_correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        train_acc = train_correct / len(train_ds)
        val_acc = val_correct / len(val_ds)
        print(f"  epoch {epoch+1}/{epochs}  train_loss={train_loss/len(train_ds):.4f}  "
              f"train_acc={train_acc*100:.1f}%  val_acc={val_acc*100:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"[RESULT] {fruit_type}: best validation accuracy = {best_val_acc*100:.1f}%")
    return model, classes


def save_cnn_model(model, classes, path):
    import torch
    torch.save({"state_dict": model.state_dict(), "classes": classes}, path)
    print(f"[INFO] Saved to {path}")


def load_cnn_model(path):
    import torch
    checkpoint = torch.load(path, map_location="cpu")
    classes = checkpoint["classes"]
    model = _build_model(len(classes))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, classes


def predict_quality(model, classes, image_or_path):
    """
    image_or_path: a file path, a PIL Image, or a BGR numpy array
    (OpenCV convention — auto-converted to RGB). Returns (label, confidence).
    """
    import torch
    import numpy as np
    from PIL import Image

    if isinstance(image_or_path, str):
        image = Image.open(image_or_path).convert("RGB")
    elif isinstance(image_or_path, np.ndarray):
        image = Image.fromarray(image_or_path[:, :, ::-1])  # BGR -> RGB
    else:
        image = image_or_path

    transform = _build_transforms(train=False)
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        idx = int(probs.argmax())
        return classes[idx], float(probs[idx])


def main():
    parser = argparse.ArgumentParser(description="Train CNN-based fruit quality classifiers (one per fruit type).")
    parser.add_argument("--dataset", type=str, default="dataset/train", help="Path to training image folder")
    parser.add_argument("--out", type=str, default="cnn_quality_models", help="Output directory for trained models")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    fruit_types = sorted({ftype for _, (ftype, _) in CLASS_FOLDERS.items()})

    for fruit_type in fruit_types:
        model, classes = train_fruit_type_model(args.dataset, fruit_type, epochs=args.epochs, batch_size=args.batch_size)
        if model is not None:
            save_cnn_model(model, classes, os.path.join(args.out, f"{fruit_type}.pt"))


if __name__ == "__main__":
    main()
