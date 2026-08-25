
import glob
import os

from train_cnn_quality import (
    CLASS_FOLDERS,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    VAL_SPLIT,
    _base_image_key,
    _group_split,
    _build_model,
    _build_transforms,
    prepare_cnn_input_bgr,
    save_cnn_model,
    load_cnn_model as _load_cnn_model_impl,
    _predict_quality_impl,
)


class FruitTypeDataset:

    def __init__(self, dataset_dir, transform=None):
        self.transform = transform
        self.samples = []
        self.classes = sorted({ftype for _, (ftype, _quality) in CLASS_FOLDERS.items()})
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for folder, (ftype, _quality) in CLASS_FOLDERS.items():
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
                self.samples.append((p, self.class_to_idx[ftype], _base_image_key(p)))

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


def train_type_classifier(dataset_dir, epochs=EPOCHS, batch_size=BATCH_SIZE):
    import torch
    from torch.utils.data import DataLoader, Subset
    import torch.nn as nn
    import torch.optim as optim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training fruit-TYPE model on device: {device}")

    full_dataset = FruitTypeDataset(dataset_dir, transform=_build_transforms(train=True))
    if len(full_dataset) == 0:
        print("[WARN] No images found for any fruit type, aborting.")
        return None, None

    classes = full_dataset.classes
    print(f"[INFO] Fruit type: {len(full_dataset)} images, classes={classes}")

    train_indices, val_indices = _group_split(full_dataset, VAL_SPLIT, seed=42)

    eval_dataset = FruitTypeDataset(dataset_dir, transform=_build_transforms(train=False))
    train_ds = Subset(full_dataset, train_indices)
    val_ds = Subset(eval_dataset, val_indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = _build_model(len(classes)).to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_correct, train_total, train_loss_sum = 0, 0, 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += images.size(0)

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += images.size(0)

        train_acc = 100.0 * train_correct / max(train_total, 1)
        val_acc = 100.0 * val_correct / max(val_total, 1)
        train_loss = train_loss_sum / max(train_total, 1)
        print(f"  epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  train_acc={train_acc:.1f}%  val_acc={val_acc:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"[RESULT] Fruit type: best validation accuracy = {best_val_acc:.1f}%")
    return model, classes


def load_type_model(path):
    return _load_cnn_model_impl(path)


def predict_fruit_type(model, classes, image_or_path):
    label, confidence, _probs = _predict_quality_impl(model, classes, image_or_path)
    return label, confidence


def predict_fruit_type_with_probs(model, classes, image_or_path):
    return _predict_quality_impl(model, classes, image_or_path)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train the CNN fruit-TYPE classifier (Mango vs Strawberry).")
    parser.add_argument("--dataset", type=str, default="dataset/train", help="Path to training image folder")
    parser.add_argument("--out", type=str, default="cnn_type_models", help="Output directory for the trained model")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model, classes = train_type_classifier(args.dataset, epochs=args.epochs, batch_size=args.batch_size)
    if model is not None:
        save_cnn_model(model, classes, os.path.join(args.out, "fruit_type.pt"))


if __name__ == "__main__":
    main()
