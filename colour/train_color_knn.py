import glob
import json
import os
import random
import re
from collections import Counter, defaultdict
from multiprocessing import Pool

import cv2
import joblib
import numpy as np
from scipy.stats import skew
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from segmentation import segmentation_mask_and_contour


COLOR_KNN_MODELS_DIR = "color_knn_models"
DATASET_DIR = "dataset/train"
REPORT_PATH = "report_assets/color_knn_summary.json"

VAL_SPLIT = float(os.environ.get("COLOR_KNN_VAL_SPLIT", "0.2"))
SEED = int(os.environ.get("COLOR_KNN_SEED", "42"))
MAX_PER_CLASS = int(os.environ.get("COLOR_KNN_MAX_PER_CLASS", "0")) or None
MAX_IMAGE_SIDE = int(os.environ.get("COLOR_KNN_MAX_IMAGE_SIDE", "500"))

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

_AUG_PREFIX_RE = re.compile(
    r"^(rotated_by_\d+_|translation_|saltandpepper_|salt_and_pepper_|"
    r"vertical_flip_|horizontal_flip_|aug_)+",
    re.IGNORECASE,
)


def _base_image_key(path):
    """Map an original image and its augmented copies to one group."""
    stem = os.path.splitext(os.path.basename(path))[0]
    previous = None
    while previous != stem:
        previous = stem
        stem = _AUG_PREFIX_RE.sub("", stem)
    return stem.lower()


def _normalised_hist(values, bins, value_range):
    hist, _ = np.histogram(values, bins=bins, range=value_range)
    hist = hist.astype(np.float32)
    total = float(hist.sum())
    if total > 0:
        hist /= total
    return hist.tolist()


def _channel_statistics(values):
    """Robust distribution features for one colour channel."""
    values = values.astype(np.float32)
    std = float(values.std())
    result = [
        float(values.mean()),
        std,
        float(skew(values)) if std > 1e-6 else 0.0,
    ]
    result.extend(float(x) for x in np.percentile(values, [10, 25, 50, 75, 90]))
    return result


def extract_color_features(bgr):
    """
    Extract illumination-tolerant colour/distribution/spot features.

    The returned vector has a fixed length. Only foreground pixels from the
    segmentation mask are used, so the background does not dominate KNN.
    """
    mask, contour = segmentation_mask_and_contour(bgr)
    if contour is None or mask is None:
        return None

    mask = (mask > 0).astype(np.uint8)
    if int(mask.sum()) < 100:
        return None

    # Remove a thin boundary that often contains background pixels.
    h, w = mask.shape[:2]
    kernel_size = max(3, int(round(min(h, w) * 0.008)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = min(kernel_size, 7)
    eroded = cv2.erode(mask, np.ones((kernel_size, kernel_size), np.uint8))
    if int(eroded.sum()) >= 100:
        mask = eroded

    fg = mask.astype(bool)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    features = []

    # LAB and HSV robust channel statistics: 6 channels x 8 values.
    for colour_space in (lab, hsv):
        for channel in range(3):
            features.extend(_channel_statistics(colour_space[:, :, channel][fg]))

    # Hue is circular: 0 and 179 are close colours, not far-apart numbers.
    hue_radians = hsv[:, :, 0][fg] * (2.0 * np.pi / 180.0)
    features.extend([
        float(np.sin(hue_radians).mean()),
        float(np.cos(hue_radians).mean()),
    ])

    # Histograms retain multimodal colour information that moments lose.
    features.extend(_normalised_hist(hsv[:, :, 0][fg], 18, (0, 180)))
    features.extend(_normalised_hist(hsv[:, :, 1][fg], 10, (0, 256)))
    features.extend(_normalised_hist(lab[:, :, 1][fg], 12, (0, 256)))
    features.extend(_normalised_hist(lab[:, :, 2][fg], 12, (0, 256)))

    hue = hsv[:, :, 0][fg]
    saturation = hsv[:, :, 1][fg]
    value = hsv[:, :, 2][fg]

    chromatic = saturation >= 45
    denominator = max(1, int(fg.sum()))

    # OpenCV hue ranges: red wraps around 0, green ~60, yellow ~30.
    colour_ratios = [
        np.count_nonzero(chromatic & (hue >= 35) & (hue < 90)) / denominator,
        np.count_nonzero(chromatic & (hue >= 18) & (hue < 35)) / denominator,
        np.count_nonzero(chromatic & ((hue < 12) | (hue >= 170))) / denominator,
        np.count_nonzero(value < 55) / denominator,
        np.count_nonzero(value > 225) / denominator,
        np.count_nonzero(saturation < 35) / denominator,
    ]
    features.extend(float(x) for x in colour_ratios)

    # Simple texture/spot cues help separate fresh fruit from rotten patches.
    lightness = lab[:, :, 0]
    laplacian = cv2.Laplacian(lightness, cv2.CV_32F, ksize=3)
    sobel_x = cv2.Sobel(lightness, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(lightness, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(sobel_x, sobel_y)
    features.extend([
        float(np.mean(np.abs(laplacian[fg]))),
        float(np.std(laplacian[fg])),
        float(np.mean(gradient[fg])),
        float(np.percentile(gradient[fg], 90)),
    ])

    vector = np.asarray(features, dtype=np.float32)
    if not np.all(np.isfinite(vector)):
        return None
    return vector


# Backwards-compatible name if another module imports the old function.
extract_color_moments = extract_color_features


def save_color_knn_model(scaler, model, classes, path, metadata=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(
        {
            "scaler": scaler,
            "model": model,
            "classes": classes,
            "metadata": metadata or {},
        },
        path,
    )
    print(f"[INFO] Saved colour KNN model to {path}")


def load_color_knn_model(path):
    payload = joblib.load(path)
    return payload["scaler"], payload["model"], payload["classes"]


def predict_ripeness_from_color(bgr, scaler, model, classes):
    features = extract_color_features(bgr)
    if features is None:
        return None, 0.0, {}

    features_scaled = scaler.transform(features.reshape(1, -1))
    numeric_label = int(model.predict(features_scaled)[0])

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features_scaled)[0]
        # predict_proba columns follow model.classes_, which is safer than
        # assuming that every numeric class is present in the fitted model.
        probability_dict = {
            classes[int(class_id)]: float(probability)
            for class_id, probability in zip(model.classes_, probabilities)
        }
        predicted_class = classes[numeric_label]
        return predicted_class, probability_dict[predicted_class], probability_dict

    return classes[numeric_label], 1.0, {classes[numeric_label]: 1.0}


def _extract_one(args):
    path, label, group_key = args
    bgr = cv2.imread(path)
    if bgr is None:
        return None

    height, width = bgr.shape[:2]
    if max(height, width) > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / float(max(height, width))
        bgr = cv2.resize(
            bgr,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    features = extract_color_features(bgr)
    if features is None:
        return None
    return features, label, group_key


def stratified_group_holdout(y, groups, val_fraction=VAL_SPLIT, seed=SEED):
    """
    Split whole augmentation groups while preserving each class in both sets.

    Each group is required to contain only one class. The group key built in
    main() includes the class name to avoid accidental filename collisions.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("VAL_SPLIT must be between 0 and 1")

    group_indices = defaultdict(list)
    for index, group in enumerate(groups):
        group_indices[group].append(index)

    groups_by_class = defaultdict(list)
    for group, indices in group_indices.items():
        labels = {int(y[index]) for index in indices}
        if len(labels) != 1:
            raise ValueError(f"Group {group!r} contains multiple labels: {labels}")
        groups_by_class[labels.pop()].append(group)

    rng = random.Random(seed)
    validation_groups = set()

    for class_id, class_groups in groups_by_class.items():
        class_groups = list(class_groups)
        rng.shuffle(class_groups)
        if len(class_groups) < 2:
            raise ValueError(
                f"Class {class_id} has only {len(class_groups)} original-image group(s); "
                "at least 2 are needed for an honest holdout."
            )

        class_image_count = sum(len(group_indices[g]) for g in class_groups)
        target_images = max(1, round(class_image_count * val_fraction))
        chosen = []
        chosen_images = 0

        # Keep at least one original-image group in training.
        for group in class_groups[:-1]:
            if not chosen or chosen_images < target_images:
                chosen.append(group)
                chosen_images += len(group_indices[group])
            else:
                break
        validation_groups.update(chosen)

    validation_indices = [
        index for index, group in enumerate(groups) if group in validation_groups
    ]
    training_indices = [
        index for index, group in enumerate(groups) if group not in validation_groups
    ]
    return np.asarray(training_indices), np.asarray(validation_indices)


def _print_class_counts(name, labels, classes):
    counts = Counter(int(x) for x in labels)
    readable = {class_name: counts.get(i, 0) for i, class_name in enumerate(classes)}
    print(f"  {name} class counts: {readable}")


def _build_parameter_grid(max_k):
    candidates = [1, 3, 5, 7, 9, 11, 15, 21, 31]
    k_values = [k for k in candidates if k <= max_k]
    if not k_values:
        k_values = [1]
    return {
        "knn__n_neighbors": k_values,
        "knn__weights": ["uniform", "distance"],
        "knn__metric": ["euclidean", "manhattan"],
    }


def tune_knn(X_train, y_train, train_groups):
    unique_groups_per_class = []
    for class_id in np.unique(y_train):
        unique_groups_per_class.append(
            len(set(train_groups[y_train == class_id].tolist()))
        )
    cv_splits = min(5, min(unique_groups_per_class))

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_jobs=-1)),
    ])

    if cv_splits < 2:
        print("  [WARN] Too few original-image groups for CV; using safe defaults")
        pipeline.set_params(
            knn__n_neighbors=min(5, len(X_train)),
            knn__weights="distance",
            knn__metric="manhattan",
        )
        pipeline.fit(X_train, y_train)
        return pipeline, None, cv_splits

    cv = StratifiedGroupKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=SEED,
    )
    cv_indices = list(cv.split(X_train, y_train, groups=train_groups))
    # Augmentation groups can have unequal sizes, so derive the safe K limit
    # from the smallest actual CV training fold instead of an approximation.
    max_k = max(1, min(len(fold_train) for fold_train, _ in cv_indices))
    parameter_grid = _build_parameter_grid(max_k)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="balanced_accuracy",
        cv=cv_indices,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
        error_score="raise",
    )
    search.fit(X_train, y_train, groups=train_groups)
    print(f"  Best CV balanced accuracy: {search.best_score_ * 100:.2f}%")
    print(f"  Best parameters: {search.best_params_}")
    return search.best_estimator_, search, cv_splits


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    fruit_types = sorted({fruit for fruit, _ in CLASS_FOLDERS.values()})
    summary = {
        "seed": SEED,
        "val_split": VAL_SPLIT,
        "max_per_class": MAX_PER_CLASS,
        "feature": (
            "LAB/HSV moments and percentiles, circular hue, colour histograms, "
            "colour ratios and luminance texture"
        ),
        "models": {},
    }

    for fruit_type in fruit_types:
        print(f"\n=== {fruit_type} ===")
        paths, labels, group_keys = [], [], []
        classes = sorted({
            quality
            for fruit, quality in CLASS_FOLDERS.values()
            if fruit == fruit_type
        })
        class_to_index = {name: index for index, name in enumerate(classes)}

        for folder, (fruit, quality) in CLASS_FOLDERS.items():
            if fruit != fruit_type:
                continue

            class_dir = os.path.join(DATASET_DIR, folder)
            folder_paths = sorted(
                glob.glob(os.path.join(class_dir, "*.jpg"))
                + glob.glob(os.path.join(class_dir, "*.jpeg"))
                + glob.glob(os.path.join(class_dir, "*.png"))
            )

            if MAX_PER_CLASS and len(folder_paths) > MAX_PER_CLASS:
                rng = random.Random(f"{SEED}:{fruit_type}:{quality}")
                folder_paths = sorted(rng.sample(folder_paths, MAX_PER_CLASS))

            for path in folder_paths:
                paths.append(path)
                labels.append(class_to_index[quality])
                # Including quality prevents unrelated files such as image1.jpg
                # in different classes from being merged into one group.
                group_keys.append(f"{quality.lower()}::{_base_image_key(path)}")

        if not paths:
            print("  [WARN] No images found; skipping")
            continue

        print(f"  {len(paths)} images selected; extracting features...")
        X, y, valid_groups = [], [], []
        work_items = zip(paths, labels, group_keys)
        with Pool(processes=os.cpu_count() or 2) as pool:
            for result in pool.imap(_extract_one, work_items, chunksize=8):
                if result is None:
                    continue
                features, label, group_key = result
                X.append(features)
                y.append(label)
                valid_groups.append(group_key)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        valid_groups = np.asarray(valid_groups, dtype=object)
        print(f"  {len(X)} usable feature vectors; {X.shape[1]} features")

        missing_classes = [
            classes[i] for i in range(len(classes)) if np.count_nonzero(y == i) == 0
        ]
        if missing_classes:
            print(f"  [WARN] Missing usable classes {missing_classes}; skipping")
            continue

        train_idx, validation_idx = stratified_group_holdout(
            y, valid_groups, VAL_SPLIT, SEED
        )
        X_train, y_train = X[train_idx], y[train_idx]
        X_validation, y_validation = X[validation_idx], y[validation_idx]
        train_groups = valid_groups[train_idx]

        _print_class_counts("Train", y_train, classes)
        _print_class_counts("Validation", y_validation, classes)

        best_pipeline, search, cv_splits = tune_knn(
            X_train, y_train, train_groups
        )
        predictions = best_pipeline.predict(X_validation)

        accuracy = accuracy_score(y_validation, predictions)
        balanced_accuracy = balanced_accuracy_score(y_validation, predictions)
        macro_f1 = f1_score(y_validation, predictions, average="macro")
        matrix = confusion_matrix(
            y_validation, predictions, labels=list(range(len(classes)))
        )
        report = classification_report(
            y_validation,
            predictions,
            labels=list(range(len(classes))),
            target_names=classes,
            output_dict=True,
            zero_division=0,
        )

        print(f"  Holdout accuracy:          {accuracy * 100:.2f}%")
        print(f"  Holdout balanced accuracy: {balanced_accuracy * 100:.2f}%")
        print(f"  Holdout macro F1:          {macro_f1 * 100:.2f}%")
        print("  Confusion matrix:")
        print(matrix)

        scaler = best_pipeline.named_steps["scaler"]
        knn = best_pipeline.named_steps["knn"]
        best_params = (
            search.best_params_
            if search is not None
            else {
                "knn__n_neighbors": knn.n_neighbors,
                "knn__weights": knn.weights,
                "knn__metric": knn.metric,
            }
        )

        model_result = {
            "classes": classes,
            "train_n": int(len(X_train)),
            "validation_n": int(len(X_validation)),
            "feature_count": int(X.shape[1]),
            "cv_splits": int(cv_splits),
            "best_params": best_params,
            "cv_balanced_accuracy": (
                float(search.best_score_) if search is not None else None
            ),
            "accuracy": float(accuracy),
            "balanced_accuracy": float(balanced_accuracy),
            "macro_f1": float(macro_f1),
            "confusion_matrix": matrix.tolist(),
            "classification_report": report,
        }

        model_path = os.path.join(
            COLOR_KNN_MODELS_DIR, f"{fruit_type}.joblib"
        )
        save_color_knn_model(
            scaler,
            knn,
            classes,
            model_path,
            metadata=model_result,
        )
        summary["models"][fruit_type.lower()] = model_result

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as report_file:
        json.dump(summary, report_file, indent=2, ensure_ascii=False)
    print(f"\n[DONE] {REPORT_PATH} written")


if __name__ == "__main__":
    main()
