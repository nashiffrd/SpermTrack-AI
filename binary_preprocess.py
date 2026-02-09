import cv2
import os
import numpy as np
import re


KERNEL_OPEN  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
KERNEL_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
KERNEL_ERODE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))


def preprocess_morphology_binary(
    roi_dir,
    output_dir
):
    """
    Binary + erosi + isolasi sperma target (center-based)
    Output: sperma solid (hitam), background putih
    """

    os.makedirs(output_dir, exist_ok=True)

    saved = 0

    for fname in os.listdir(roi_dir):
        if not fname.endswith(".png"):
            continue

        img_path = os.path.join(roi_dir, fname)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # adaptive threshold
        binary = cv2.adaptiveThreshold(
            img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )

        # erosi + opening
        binary = cv2.erode(binary, KERNEL_ERODE, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, KERNEL_OPEN)

        # connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

        h, w = img.shape
        cx_img, cy_img = w // 2, h // 2

        min_dist = np.inf
        target_label = None

        for i in range(1, num_labels):
            cx, cy = centroids[i]
            dist = np.sqrt((cx - cx_img)**2 + (cy - cy_img)**2)
            if dist < min_dist:
                min_dist = dist
                target_label = i

        if target_label is None:
            continue

        mask = np.zeros_like(binary)
        mask[labels == target_label] = 255

        # fill contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, contours, -1, 255, thickness=-1)

        # rapikan
        filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, KERNEL_CLOSE)

        # invert
        final = 255 - filled

        cv2.imwrite(os.path.join(output_dir, fname), final)
        saved += 1

    return saved
