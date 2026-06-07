"""
MODULE 2 — PaddleOCR Engine
Strong alternative to Tesseract — better on curved/noisy handwriting.
Install: pip install paddlepaddle paddleocr
"""

import numpy as np
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    filename    : str
    raw_text    : str
    cleaned_text: str
    confidence  : float
    engine      : str = "paddle"
    error       : Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return self.error is None and len(self.cleaned_text) > 0


class PaddleOCREngine:
    """
    PaddleOCR engine — fast, accurate, supports multiple languages.
    Install: pip install paddlepaddle paddleocr
    """

    def __init__(self, lang: str = "en", use_gpu: bool = False,
                 use_angle_cls: bool = True):
        """
        Args:
            lang          : language code ('en', 'ch', 'hi' etc.)
            use_gpu       : True if CUDA GPU available
            use_angle_cls : correct 180° rotated text
        """
        self.lang          = lang
        self.use_gpu       = use_gpu
        self.use_angle_cls = use_angle_cls
        self._ocr          = None
        self._loaded       = False

    def load(self):
        """Lazy-load PaddleOCR model."""
        try:
            from paddleocr import PaddleOCR
            self._ocr    = PaddleOCR(
                use_angle_cls=self.use_angle_cls,
                lang=self.lang,
                use_gpu=self.use_gpu,
                show_log=False
            )
            self._loaded = True
            logger.info("PaddleOCR loaded ✅ (lang=%s, gpu=%s)", self.lang, self.use_gpu)
        except ImportError:
            raise RuntimeError(
                "PaddleOCR not installed.\n"
                "Fix: pip install paddlepaddle paddleocr\n"
                "For GPU: pip install paddlepaddle-gpu paddleocr"
            )

    # ── PUBLIC ──────────────────────────────────────────────────────────
    def extract(self, image_input, filename: str = "unknown") -> OCRResult:
        """Extract text from a single image."""
        if not self._loaded:
            self.load()
        try:
            img_array = self._to_numpy(image_input)
            result    = self._ocr.ocr(img_array, cls=self.use_angle_cls)

            if not result or not result[0]:
                return OCRResult(filename=filename, raw_text="", cleaned_text="",
                                 confidence=0.0, error="No text detected")

            lines      = result[0]
            texts      = [line[1][0] for line in lines]
            confs      = [line[1][1] for line in lines]
            raw_text   = " ".join(texts)
            avg_conf   = float(np.mean(confs))
            cleaned    = self._clean_text(raw_text)

            return OCRResult(filename=filename, raw_text=raw_text,
                             cleaned_text=cleaned, confidence=avg_conf)
        except Exception as e:
            logger.error("PaddleOCR failed on %s: %s", filename, e)
            return OCRResult(filename=filename, raw_text="", cleaned_text="",
                             confidence=0.0, error=str(e))

    def extract_batch(self, image_list: list) -> list:
        """Process list of (filename, image) tuples."""
        return [self.extract(img, filename=fname) for fname, img in image_list]

    # ── HELPERS ─────────────────────────────────────────────────────────
    def _to_numpy(self, image_input) -> np.ndarray:
        import cv2
        from PIL import Image as PILImage
        if isinstance(image_input, np.ndarray):
            # PaddleOCR needs BGR 3-channel
            if len(image_input.shape) == 2:
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            return image_input
        if isinstance(image_input, PILImage.Image):
            arr = np.array(image_input.convert("RGB"))
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input))
            if img is None:
                raise FileNotFoundError(f"Not found: {image_input}")
            return img
        raise TypeError(f"Unsupported type: {type(image_input)}")

    @staticmethod
    def _clean_text(text: str) -> str:
        import re
        return re.sub(r"[^A-Za-z0-9\s]", "", text).strip().upper()


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: python paddle_engine.py <image_path>")
        sys.exit(1)
    engine = PaddleOCREngine()
    result = engine.extract(sys.argv[1], filename=Path(sys.argv[1]).name)
    print(f"Extracted : '{result.cleaned_text}'")
    print(f"Confidence: {result.confidence:.2f}")
