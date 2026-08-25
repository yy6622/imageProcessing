import argparse
import glob
import json
import os
import random
from collections import Counter, defaultdict

import cv2
import numpy as np

from train_cnn_quality import CLASS_FOLDERS, _base_image_key, _build_model


TYPE_IMAGE_SIZE = int(os.environ.get("TYPE_CNN_IMAGE_SIZE", "224"))
TYPE_BATCH_SIZE = int(os.environ.get("TYPE_CNN_BATCH_SIZE", "16"))
TYPE_EPOCHS = int(os.environ.get("TYPE_CNN_EPOCHS", "30"))
TYPE_LEARNING_RATE = float(os.environ.get("TYPE_CNN_LEARNING_RATE", "0.00015"))
TYPE_WEIGHT_DECAY = float(os.environ.get("TYPE_CNN_WEIGHT_DECAY", "0.0001"))
TYPE_VAL_SPLIT = float(os.environ.get("TYPE_CNN_VAL_SPLIT", "0.2"))
TYPE_EARLY_STOPPING = int(os.environ.get("TYPE_CNN_EARLY_STOPPING", "7"))
TYPE_MAX_INPUT_SIDE = int(os.environ.get("TYPE_CNN_MAX_INPUT_SIDE", "640"))
SEED = int(os.environ.get("TYPE_CNN_SEED", "42"))


def _set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _letterbox_bgr(image, size=TYPE_IMAGE_SIZE, fill=(0, 0, 0)):
    """Resize without destroying the fruit's aspect ratio."""
    if image is None or image.size == 0:
        raise ValueError("Cannot letterbox an empty image")

    height, width = image.shape[:2]
    scale = min(size / float(width), size / float(height))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)

    canvas = np.full((size, size, 3), fill, dtype=np.uint8)
    x = (size - new_width) // 2
    y = (size - new_height) // 2
    canvas[y:y + new_height, x:x + new_width] = resized
    return canvas


def prepare_type_input_bgr(raw_bgr, mask_background=True):
    """
    Prepare fruit-type input identically for training and inference.

    Unlike the previous implementation, this crops then letterboxes the fruit,
    preserving the elongated Mango/Banana shape that separates it from Orange.
    """
    import preprocessing as prep
    from segmentation import segmentation_mask_and_contour

    if raw_bgr is None or raw_bgr.size == 0:
        raise ValueError("raw_bgr must be a non-empty BGR image")

    bgr = raw_bgr.copy()
    height, width = bgr.shape[:2]
    if max(height, width) > TYPE_MAX_INPUT_SIDE:
        scale = TYPE_MAX_INPUT_SIDE / float(max(height, width))
        bgr = cv2.resize(
            bgr,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    # The same mild preprocessing is used in both training and prediction.
    bgr = prep.preprocess_image(
        bgr,
        denoise_method="median",
        enhance_method="clahe",
    )

    if mask_background:
        mask, contour = segmentation_mask_and_contour(bgr)
        if contour is not None:
            x, y, w, h = cv2.boundingRect(contour)
            pad = max(2, int(round(max(w, h) * 0.06)))
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1 = min(bgr.shape[1], x + w + pad)
            y1 = min(bgr.shape[0], y + h + pad)
            isolated = cv2.bitwise_and(bgr, bgr, mask=mask)
            bgr = isolated[y0:y1, x0:x1]

    return _letterbox_bgr(bgr, TYPE_IMAGE_SIZE)


def _build_type_transforms(train):
    from torchvision import transforms

    operations = []
    if train:
        # Modest augmentation: retain colour and silhouette information.
        operations.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(12),
            transforms.ColorJitter(
                brightness=0.18,
                contrast=0.18,
                saturation=0.12,
                hue=0.02,
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.04, 0.04),
                scale=(0.92, 1.08),
            ),
        ])
    operations.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    if train:
        operations.append(transforms.RandomErasing(p=0.10, scale=(0.01, 0.04)))
    return transforms.Compose(operations)


class FruitTypeDataset:
    def __init__(self, dataset_dir, transform=None):
        self.transform = transform
        self.samples = []
        self.classes = sorted({fruit for fruit, _quality in CLASS_FOLDERS.values()})
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}

        for folder, (fruit_type, quality) in CLASS_FOLDERS.items():
            class_dir = os.path.join(dataset_dir, folder)
            if not os.path.isdir(class_dir):
                print(f"[WARN] Folder not found, skipping: {class_dir}")
                continue

            paths = sorted(
                glob.glob(os.path.join(class_dir, "*.jpg"))
                + glob.glob(os.path.join(class_dir, "*.jpeg"))
                + glob.glob(os.path.join(class_dir, "*.png"))
            )
            for path in paths:
                # Prefix with fruit and quality so image1.jpg in unrelated
                # folders cannot merge into a conflicting-label group.
                group = (
                    f"{fruit_type.lower()}::{quality.lower()}::"
                    f"{_base_image_key(path)}"
                )
                self.samples.append((path, self.class_to_idx[fruit_type], group))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        from PIL import Image

        path, label, _group = self.samples[index]
        bgr = cv2.imread(path)
        if bgr is None:
            try:
                rgb = np.asarray(Image.open(path).convert("RGB"))
                bgr = rgb[:, :, ::-1].copy()
            except (FileNotFoundError, OSError) as exc:
                raise RuntimeError(f"Unreadable training image {path}: {exc}") from exc

        prepared = prepare_type_input_bgr(bgr)
        image = Image.fromarray(cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB))
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def stratified_group_split(dataset, val_fraction=TYPE_VAL_SPLIT, seed=SEED):
    """Keep augmentation families together and every class represented."""
    group_indices = defaultdict(list)
    for index, (_path, _label, group) in enumerate(dataset.samples):
        group_indices[group].append(index)

    groups_by_class = defaultdict(list)
    for group, indices in group_indices.items():
        labels = {dataset.samples[index][1] for index in indices}
        if len(labels) != 1:
            raise ValueError(f"Group {group!r} contains conflicting labels: {labels}")
        groups_by_class[labels.pop()].append(group)

    random_generator = random.Random(seed)
    validation_groups = set()
    for class_id in range(len(dataset.classes)):
        groups = list(groups_by_class.get(class_id, []))
        if len(groups) < 2:
            raise ValueError(
                f"{dataset.classes[class_id]} has only {len(groups)} original-image "
                "groups; at least 2 are required."
            )
        random_generator.shuffle(groups)
        class_images = sum(len(group_indices[group]) for group in groups)
        target = max(1, round(class_images * val_fraction))
        selected_count = 0
        for group in groups[:-1]:
            if selected_count >= target and selected_count > 0:
                break
            validation_groups.add(group)
            selected_count += len(group_indices[group])

    train_indices = [
        index
        for index, (_path, _label, group) in enumerate(dataset.samples)
        if group not in validation_groups
    ]
    validation_indices = [
        index
        for index, (_path, _label, group) in enumerate(dataset.samples)
        if group in validation_groups
    ]
    return train_indices, validation_indices


def _class_counts(dataset, indices):
    return Counter(dataset.samples[index][1] for index in indices)


def _print_counts(name, dataset, indices):
    counts = _class_counts(dataset, indices)
    readable = {
        class_name: int(counts.get(index, 0))
        for index, class_name in enumerate(dataset.classes)
    }
    print(f"[INFO] {name} counts: {readable}")


def train_type_classifier(
    dataset_dir,
    epochs=TYPE_EPOCHS,
    batch_size=TYPE_BATCH_SIZE,
):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    from torch.utils.data import DataLoader, Subset

    _set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training five-class fruit-type model on {device}")

    train_dataset = FruitTypeDataset(
        dataset_dir, transform=_build_type_transforms(train=True)
    )
    if len(train_dataset) == 0:
        raise ValueError(f"No images found below {dataset_dir}")

    evaluation_dataset = FruitTypeDataset(
        dataset_dir, transform=_build_type_transforms(train=False)
    )
    train_indices, validation_indices = stratified_group_split(train_dataset)
    _print_counts("Train", train_dataset, train_indices)
    _print_counts("Validation", train_dataset, validation_indices)

    train_subset = Subset(train_dataset, train_indices)
    validation_subset = Subset(evaluation_dataset, validation_indices)

    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    classes = train_dataset.classes
    counts = _class_counts(train_dataset, train_indices)
    class_weights = torch.tensor(
        [
            len(train_indices) / max(1, len(classes) * counts.get(i, 0))
            for i in range(len(classes))
        ],
        dtype=torch.float32,
        device=device,
    )

    model = _build_model(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=TYPE_LEARNING_RATE,
        weight_decay=TYPE_WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6
    )

    best_macro_f1 = -1.0
    best_state = None
    epochs_without_improvement = 0
    best_report = None
    best_matrix = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_correct = 0
        train_total = 0
        train_loss_sum = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * images.size(0)
            train_correct += int((outputs.argmax(1) == labels).sum().item())
            train_total += images.size(0)

        model.eval()
        validation_labels = []
        validation_predictions = []
        with torch.no_grad():
            for images, labels in validation_loader:
                outputs = model(images.to(device))
                predictions = outputs.argmax(1).cpu().numpy().tolist()
                validation_predictions.extend(predictions)
                validation_labels.extend(labels.numpy().tolist())

        validation_accuracy = float(np.mean(
            np.asarray(validation_predictions) == np.asarray(validation_labels)
        ))
        macro_f1 = f1_score(
            validation_labels,
            validation_predictions,
            labels=list(range(len(classes))),
            average="macro",
            zero_division=0,
        )
        scheduler.step(macro_f1)
        learning_rate = optimizer.param_groups[0]["lr"]
        print(
            f"  epoch {epoch}/{epochs} "
            f"loss={train_loss_sum / max(1, train_total):.4f} "
            f"train_acc={100 * train_correct / max(1, train_total):.1f}% "
            f"val_acc={100 * validation_accuracy:.1f}% "
            f"macro_f1={100 * macro_f1:.1f}% lr={learning_rate:.2e}"
        )

        if macro_f1 > best_macro_f1 + 1e-5:
            best_macro_f1 = macro_f1
            epochs_without_improvement = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_matrix = confusion_matrix(
                validation_labels,
                validation_predictions,
                labels=list(range(len(classes))),
            )
            best_report = classification_report(
                validation_labels,
                validation_predictions,
                labels=list(range(len(classes))),
                target_names=classes,
                output_dict=True,
                zero_division=0,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= TYPE_EARLY_STOPPING:
            print(f"[INFO] Early stopping after epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    model.to(device).eval()

    print(f"[RESULT] Best validation macro F1: {best_macro_f1 * 100:.2f}%")
    print("[RESULT] Confusion matrix (rows=true, columns=predicted):")
    print("          " + " ".join(f"{name[:6]:>6}" for name in classes))
    for name, row in zip(classes, best_matrix):
        print(f"{name[:9]:>9} " + " ".join(f"{int(value):6d}" for value in row))
    return model, classes, best_report, best_matrix.tolist()


def save_type_model(model, classes, path, report=None, confusion=None):
    import torch

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "classes": list(classes),
            "image_size": TYPE_IMAGE_SIZE,
            "preprocessing": "segmentation + CLAHE + aspect-preserving letterbox",
        },
        path,
    )
    if report is not None:
        report_path = os.path.splitext(path)[0] + "_report.json"
        with open(report_path, "w", encoding="utf-8") as report_file:
            json.dump(
                {"classes": classes, "classification_report": report,
                 "confusion_matrix": confusion},
                report_file,
                indent=2,
            )
    print(f"[INFO] Saved fruit-type model to {path}")


def load_type_model(path):
    import torch

    checkpoint = torch.load(path, map_location="cpu")
    classes = list(checkpoint["classes"])
    model = _build_model(len(classes))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, classes


def _read_bgr(image_or_path):
    from PIL import Image

    if isinstance(image_or_path, (str, os.PathLike)):
        bgr = cv2.imread(str(image_or_path))
        if bgr is None:
            rgb = np.asarray(Image.open(image_or_path).convert("RGB"))
            bgr = rgb[:, :, ::-1].copy()
        return bgr
    if isinstance(image_or_path, np.ndarray):
        return image_or_path
    rgb = np.asarray(image_or_path.convert("RGB"))
    return rgb[:, :, ::-1].copy()


def predict_fruit_type_with_probs(model, classes, image_or_path):
    import torch
    from PIL import Image

    bgr = _read_bgr(image_or_path)
    prepared = prepare_type_input_bgr(bgr)
    image = Image.fromarray(cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB))
    tensor = _build_type_transforms(train=False)(image).unsqueeze(0)
    device = next(model.parameters()).device
    tensor = tensor.to(device)

    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    index = int(np.argmax(probabilities))
    probability_dict = {
        classes[i]: float(probabilities[i]) for i in range(len(classes))
    }
    return classes[index], float(probabilities[index]), probability_dict


def predict_fruit_type(model, classes, image_or_path):
    label, confidence, _probabilities = predict_fruit_type_with_probs(
        model, classes, image_or_path
    )
    return label, confidence


def main():
    parser = argparse.ArgumentParser(
        description="Train a five-class Apple/Banana/Mango/Orange/Strawberry classifier"
    )
    parser.add_argument("--dataset", default="dataset/train")
    parser.add_argument("--out", default="cnn_type_models")
    parser.add_argument("--epochs", type=int, default=TYPE_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=TYPE_BATCH_SIZE)
    args = parser.parse_args()

    model, classes, report, matrix = train_type_classifier(
        args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    save_type_model(
        model,
        classes,
        os.path.join(args.out, "fruit_type.pt"),
        report,
        matrix,
    )


if __name__ == "__main__":
    main()
