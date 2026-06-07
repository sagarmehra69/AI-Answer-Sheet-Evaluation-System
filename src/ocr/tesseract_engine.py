"""
MODULE 2 — Tesseract OCR Engine
Wraps pytesseract for handwritten name/word recognition.
Tuned for your dataset: FILENAME → IDENTITY (e.g. TEST_0001.jpg → KEVIN)
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """Standardized OCR output across all engines."""
    filename   : str
    raw_text   : str          # raw OCR output
    cleaned_text: str         # uppercase, stripped
    confidence : float        # 0.0 – 1.0
    engine     : str = "tesseract"
    error      : Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return self.error is None and len(self.cleaned_text) > 0


class TesseractEngine:
    """
    Tesseract OCR engine optimized for handwritten word/name images.

    Config choices:
        --psm 8  → single word mode (best for isolated name images)
        --psm 7  → single text line
        --psm 6  → uniform block of text (for paragraphs)
    """

    # PSM modes for different use cases
    PSM_SINGLE_WORD = "--psm 8 --oem 3"
    PSM_SINGLE_LINE = "--psm 7 --oem 3"
    PSM_BLOCK       = "--psm 6 --oem 3"

    def __init__(self, psm_mode: str = None, lang: str = "eng",
                 tesseract_cmd: Optional[str] = None):
        """
        Args:
            psm_mode       : Tesseract page segmentation config string
            lang           : OCR language (default: English)
            tesseract_cmd  : path to tesseract binary (if not in PATH)
        """
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        self.lang    = lang
        self.config  = psm_mode or self.PSM_SINGLE_WORD
        self._verify_installation()

    def _verify_installation(self):
        try:
            version = pytesseract.get_tesseract_version()
            logger.info("Tesseract version: %s", version)
        except Exception as e:
            raise RuntimeError(
                f"Tesseract not found: {e}\n"
                "Install: sudo apt install tesseract-ocr  (Linux)\n"
                "         brew install tesseract           (Mac)\n"
                "         https://github.com/UB-Mannheim/tesseract (Windows)"
            )

    # ── PUBLIC ──────────────────────────────────────────────────────────
    def extract(self, image_input, filename: str = "unknown") -> OCRResult:
        """
        Extract text from a single image.

        Args:
            image_input : numpy array (grayscale) | PIL Image | file path
            filename    : used for logging/tracking

        Returns:
            OCRResult with extracted text and confidence
        """
        try:
            pil_img = self._to_pil(image_input)
            raw     = pytesseract.image_to_string(
                pil_img, lang=self.lang, config=self.config
            ).strip()
            conf    = self._get_confidence(pil_img)
            cleaned = self._clean_text(raw)
            return OCRResult(filename=filename, raw_text=raw,
                             cleaned_text=cleaned, confidence=conf)
        except Exception as e:
            logger.error("Tesseract failed on %s: %s", filename, e)
            return OCRResult(filename=filename, raw_text="", cleaned_text="",
                             confidence=0.0, error=str(e))

    def extract_with_details(self, image_input, filename: str = "unknown") -> dict:
        """
        Returns word-level bounding boxes and confidences.
        Useful for debugging and visualization.
        """
        pil_img = self._to_pil(image_input)
        data    = pytesseract.image_to_data(
            pil_img, lang=self.lang, config=self.config,
            output_type=pytesseract.Output.DICT
        )
        words = []
        for i, word in enumerate(data["text"]):
            word = word.strip()
            if not word:
                continue
            conf = int(data["conf"][i])
            if conf < 0:
                continue
            words.append({
                "word"      : word,
                "confidence": conf / 100.0,
                "x"         : data["left"][i],
                "y"         : data["top"][i],
                "w"         : data["width"][i],
                "h"         : data["height"][i],
            })
        full_text = " ".join(w["word"] for w in words)
        avg_conf  = np.mean([w["confidence"] for w in words]) if words else 0.0
        return {
            "filename" : filename,
            "text"     : self._clean_text(full_text),
            "confidence": avg_conf,
            "words"    : words,
            "engine"   : "tesseract"
        }

    def extract_batch(self, image_list: list) -> list:
        """
        Process list of (filename, image) tuples.
        Returns list of OCRResult.
        """
        results = []
        for filename, img in image_list:
            result = self.extract(img, filename=filename)
            results.append(result)
            logger.debug("[%s] → '%s' (conf=%.2f)",
                         filename, result.cleaned_text, result.confidence)
        return results

    # ── HELPERS ─────────────────────────────────────────────────────────
    def _to_pil(self, image_input) -> Image.Image:
        if isinstance(image_input, Image.Image):
            return image_input
        if isinstance(image_input, np.ndarray):
            return Image.fromarray(image_input)
        if isinstance(image_input, (str, Path)):
            return Image.open(str(image_input)).convert("L")
        raise TypeError(f"Unsupported type: {type(image_input)}")

    def _get_confidence(self, pil_img: Image.Image) -> float:
        """Get mean word confidence from Tesseract."""
        try:
            data  = pytesseract.image_to_data(
                pil_img, config=self.config,
                output_type=pytesseract.Output.DICT
            )
            confs = [c for c in data["conf"] if int(c) > 0]
            return float(np.mean(confs)) / 100.0 if confs else 0.0
        except Exception:
            return 0.5

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalize OCR output — uppercase, remove non-alpha chars."""
        import re
        cleaned = re.sub(r"[^A-Za-z0-9\s]", "", text)
        return cleaned.strip().upper()


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG)
    if len(sys.argv) < 2:
        print("Usage: python tesseract_engine.py <image_path>")
        sys.exit(1)
    engine = TesseractEngine()
    result = engine.extract(sys.argv[1], filename=Path(sys.argv[1]).name)
    print(f"Extracted : '{result.cleaned_text}'")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Valid     : {result.is_valid}")
