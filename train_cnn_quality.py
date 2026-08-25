
import argparse
import glob
import os
import random
import re

CLASS_FOLDERS = {
    "freshapples":       ("Apple", "Fresh"),
    "rottenapples":      ("Apple", "Rotten"),
    "unripe apple":      ("Apple", "Unripe"),
    "freshbanana":       ("Banana", "Fresh"),
    "rottenbanana":      ("Banana", "Rotten"),
    "unripe banana":     ("Banana", "Unripe"),
    "freshoranges":      ("Orange", "Fresh"),
    "rottenoranges":     ("Orange", "Rotten"),
    "unripe orange":     ("Orange", "Unripe"),
    "ripemango":         ("Mango", "Fresh"),
    "rottonmango":       ("Mango", "Rotten"),
    "unripemango":       ("Mango", "Unripe"),
    "FreshStrawberry":   ("Strawberry", "Fresh"),
    "RottenStrawberry":  ("Strawberry", "Rotten"),
    "unripe strawberry": ("Strawberry", "Unripe"),
}

IMAGE_SIZE = 160          # CNN input resolution — MobileNetV2's native
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.2

TRAIN_DENOISE_METHOD = "median"
TRAIN_ENHANCE_METHOD = "clahe"


def prepare_cnn_input_bgr(raw_bgr, denoise_method=TRAIN_DENOISE_METHOD, enhance_method=TRAIN_ENHANCE_METHOD, mask_background=True):
    import cv2
    import preprocessing as prep
    from segmentation import segmentation_mask_and_contour

    bgr = cv2.resize(raw_bgr, (IMAGE_SIZE, IMAGE_SIZE))
    bgr = prep.preprocess_image(bgr, denoise_method=denoise_method, enhance_method=enhance_method)
    if mask_background:
        mask, contour = segmentation_mask_and_contour(bgr)
        if contour is not None:
            bgr = cv2.bitwise_and(bgr, bgr, mask=mask)
    return bgr


_AUG_PREFIX_RE = re.compile(
    r"^(rotated_by_\d+_|translation_|saltandpepper_|salt_and_pepper_|vertical_flip_|horizontal_flip_|aug_)+",
    re.IGNORECASE,
)


def _base_image_key(path):
    name = os.path.basename(path)
    stem, _ext = os.path.splitext(name)
    prev = None
    while prev != stem:
        prev = stem
        stem = _AUG_PREFIX_RE.sub("", stem)
    return stem.lower()


def _group_split(dataset, val_frac, seed=42):
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
            transforms.RandomGrayscale(p=0.15),
            transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.3, hue=0.06),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.85, 1.15), shear=5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.08)),
        ])
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class FruitQualityDataset:

    def __init__(self, dataset_dir, fruit_type, transform=None):
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
        import cv2
        import numpy as np
        from PIL import Image

        path, label, _key = self.samples[idx]
        bgr = cv2.imread(path)
        if bgr is None:
            try:
                pil_fallback = Image.open(path).convert("RGB")
                bgr = np.array(pil_fallback)[:, :, ::-1].copy()
            except (FileNotFoundError, OSError) as exc:
                print(f"[WARN] Skipping unreadable training image: {path} ({exc})")
                fallback_idx = (idx + 1) % len(self.samples)
                return self.__getitem__(fallback_idx)

        bgr = prepare_cnn_input_bgr(bgr)

        image = Image.fromarray(bgr[:, :, ::-1])  # BGR -> RGB

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

    train_indices, val_indices = _group_split(full_dataset, VAL_SPLIT, seed=42)

    eval_dataset = FruitQualityDataset(dataset_dir, fruit_type, transform=_build_transforms(train=False))
    train_ds = Subset(full_dataset, train_indices)
    val_ds = Subset(eval_dataset, val_indices)

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


def _predict_quality_impl(model, classes, image_or_path):
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
        probs_dict = {classes[i]: float(probs[i]) for i in range(len(classes))}
        return classes[idx], float(probs[idx]), probs_dict


def predict_quality(model, classes, image_or_path):
    label, confidence, _probs = _predict_quality_impl(model, classes, image_or_path)
    return label, confidence


def predict_quality_with_probs(model, classes, image_or_path):
    return _predict_quality_impl(model, classes, image_or_path)


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
