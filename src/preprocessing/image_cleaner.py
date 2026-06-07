"""
MODULE 1 — Image Cleaner
Handles: deskew, denoise, binarize, contrast enhancement
Input : raw handwritten image (any size)
Output: cleaned numpy array ready for OCR
Dataset: handwritten name images (e.g. FILENAME=TEST_0001.jpg, IDENTITY=KEVIN)
"""

import cv2
import numpy as np
from PIL import Image
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ImageCleaner:
    """
    Production-grade image preprocessing pipeline for handwritten text images.
    Designed for handwritten name/word datasets (IAM-style).
    """

    def __init__(self, target_height: int = 64, target_width: int = 512):
        self.target_height = target_height
        self.target_width  = target_width

    # ── PUBLIC ──────────────────────────────────────────────────────────
    def clean(self, image_input, return_pil: bool = False):
        """
        Full preprocessing pipeline.
        Args:
            image_input : file path (str/Path) | numpy array | PIL Image
            return_pil  : True → PIL Image, False → numpy array
        """
        img = self._load(image_input)
        img = self._to_grayscale(img)
        img = self._remove_noise(img)
        img = self._enhance_contrast(img)
        img = self._binarize(img)
        img = self._deskew(img)
        img = self._remove_borders(img)
        img = self._resize(img)
        logger.debug("Cleaned → shape=%s", img.shape)
        return Image.fromarray(img) if return_pil else img

    def clean_batch(self, image_paths: list, return_pil: bool = False) -> list:
        """Process a list of paths. Returns list of (path, cleaned_img)."""
        results = []
        for path in image_paths:
            try:
                results.append((path, self.clean(path, return_pil=return_pil)))
            except Exception as e:
                logger.warning("Skipped %s — %s", path, e)
                results.append((path, None))
        return results

    # ── PIPELINE STEPS ──────────────────────────────────────────────────
    def _load(self, image_input) -> np.ndarray:
        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input))
            if img is None:
                raise FileNotFoundError(f"Not found: {image_input}")
            return img
        if isinstance(image_input, Image.Image):
            return np.array(image_input)
        if isinstance(image_input, np.ndarray):
            return image_input.copy()
        raise TypeError(f"Unsupported type: {type(image_input)}")

    def _to_grayscale(self, img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    def _remove_noise(self, img: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(img, (3, 3), 0)

    def _enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img)

    def _binarize(self, img: np.ndarray) -> np.ndarray:
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def _deskew(self, img: np.ndarray) -> np.ndarray:
        coords = np.column_stack(np.where(img < 128))
        if coords.shape[0] < 5:
            return img
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        if abs(angle) < 0.5:
            return img
        h, w = img.shape
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)

    def _remove_borders(self, img: np.ndarray) -> np.ndarray:
        coords = cv2.findNonZero(cv2.bitwise_not(img))
        if coords is None:
            return img
        x, y, w, h = cv2.boundingRect(coords)
        pad = 5
        x = max(0, x - pad);  y = max(0, y - pad)
        w = min(img.shape[1] - x, w + 2 * pad)
        h = min(img.shape[0] - y, h + 2 * pad)
        return img[y:y + h, x:x + w]

    def _resize(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape
        scale = min(self.target_width / w, self.target_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.full((self.target_height, self.target_width), 255, dtype=np.uint8)
        x_off = (self.target_width  - new_w) // 2
        y_off = (self.target_height - new_h) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
        return canvas


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG)
    if len(sys.argv) < 2:
        print("Usage: python image_cleaner.py <image_path>")
        sys.exit(1)
    cleaner = ImageCleaner()
    out = cleaner.clean(sys.argv[1])
    cv2.imwrite("cleaned_output.png", out)
    print(f"Shape: {out.shape}  |  Saved → cleaned_output.png")