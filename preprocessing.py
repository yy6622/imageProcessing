
import cv2
import numpy as np


# ======================================================
# Noise removal
# ======================================================
def denoise_gaussian(image, ksize=5, sigma=0):
    ksize = ksize if ksize % 2 == 1 else ksize + 1  # must be odd
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def denoise_median(image, ksize=5):
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.medianBlur(image, ksize)


def denoise_bilateral(image, d=9, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def denoise(image, method="median", ksize=5):
    method = (method or "none").lower()
    if method == "gaussian":
        return denoise_gaussian(image, ksize=ksize)
    elif method == "median":
        return denoise_median(image, ksize=ksize)
    elif method == "bilateral":
        return denoise_bilateral(image)
    elif method == "none":
        return image.copy()
    else:
        raise ValueError(f"Unknown denoise method: {method}")


# ======================================================
# Enhancement
# ======================================================
def contrast_stretch(image, lower_percentile=2, upper_percentile=98):
    result = np.empty_like(image)
    for c in range(image.shape[2]):
        channel = image[:, :, c].astype(np.float32)
        lo = np.percentile(channel, lower_percentile)
        hi = np.percentile(channel, upper_percentile)
        if hi - lo < 1e-6:
            result[:, :, c] = image[:, :, c]
            continue
        stretched = (channel - lo) * (255.0 / (hi - lo))
        result[:, :, c] = np.clip(stretched, 0, 255).astype(np.uint8)
    return result


def clahe_enhance(image, clip_limit=1.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_eq = clahe.apply(l)
    merged = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def histogram_equalize(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = cv2.equalizeHist(l)
    merged = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def enhance(image, method="clahe"):
    method = (method or "none").lower()
    if method == "contrast_stretch":
        return contrast_stretch(image)
    elif method == "clahe":
        return clahe_enhance(image)
    elif method == "histogram_eq":
        return histogram_equalize(image)
    elif method == "none":
        return image.copy()
    else:
        raise ValueError(f"Unknown enhance method: {method}")


# ======================================================
# Combined pipeline
# ======================================================
def preprocess_image(
    image,
    denoise_method="median",
    denoise_ksize=5,
    enhance_method="clahe",
):
    if image is None:
        raise ValueError("preprocess_image received None (image failed to load)")

    step1 = denoise(image, method=denoise_method, ksize=denoise_ksize)
    step2 = enhance(step1, method=enhance_method)
    return step2


def preprocess_with_steps(image, denoise_method="median", denoise_ksize=5, enhance_method="clahe"):
    original = image.copy()
    denoised = denoise(original, method=denoise_method, ksize=denoise_ksize)
    enhanced = enhance(denoised, method=enhance_method)
    return {
        "original": original,
        "denoised": denoised,
        "enhanced": enhanced,
    }