"""
MODULE 2 — TrOCR Engine (Microsoft Transformer OCR)
Deep learning OCR — significantly better than Tesseract on handwriting.
Uses HuggingFace transformers — requires: pip install transformers torch
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
    engine      : str = "trocr"
    error       : Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return self.error is None and len(self.cleaned_text) > 0


class TrOCREngine:
    """
    TrOCR — Transformer-based OCR by Microsoft.
    Model: microsoft/trocr-base-handwritten (best balance of speed/accuracy)
    Alternative: microsoft/trocr-large-handwritten (slower but more accurate)

    Install: pip install transformers torch pillow
    """

    MODEL_OPTIONS = {
        "base"  : "microsoft/trocr-base-handwritten",
        "large" : "microsoft/trocr-large-handwritten",
        "printed": "microsoft/trocr-base-printed",
    }

    def __init__(self, model_size: str = "base", device: str = "auto",
                 cache_dir: Optional[str] = None):
        """
        Args:
            model_size : 'base' | 'large' | 'printed'
            device     : 'auto' | 'cuda' | 'cpu'
            cache_dir  : local path to cache downloaded model weights
        """
        self.model_name = self.MODEL_OPTIONS.get(model_size, self.MODEL_OPTIONS["base"])
        self.cache_dir  = cache_dir
        self.device     = self._resolve_device(device)
        self.processor  = None
        self.model      = None
        self._loaded    = False

    def load(self):
        """
        Lazy-load model — call explicitly or it auto-loads on first extract().
        Downloads ~350MB on first run, cached after that.
        """
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            from PIL import Image  # noqa

            logger.info("Loading TrOCR model: %s (device=%s)", self.model_name, self.device)
            self.processor = TrOCRProcessor.from_pretrained(
                self.model_name, cache_dir=self.cache_dir
            )
            self.model = VisionEncoderDecoderModel.from_pretrained(
                self.model_name, cache_dir=self.cache_dir
            ).to(self.device)
            self.model.eval()
            self._loaded = True
            logger.info("TrOCR model loaded ✅")
        except ImportError:
            raise RuntimeError(
                "TrOCR requires: pip install transformers torch\n"
                "Or install all at once: pip install transformers torch torchvision"
            )

    # ── PUBLIC ──────────────────────────────────────────────────────────
    def extract(self, image_input, filename: str = "unknown") -> OCRResult:
        """Extract text from a single image using TrOCR."""
        if not self._loaded:
            self.load()
        try:
            import torch
            from PIL import Image as PILImage

            pil_img = self._to_pil(image_input)
            # TrOCR needs RGB
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")

            pixel_values = self.processor(
                images=pil_img, return_tensors="pt"
            ).pixel_values.to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values)

            raw_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()

            cleaned = self._clean_text(raw_text)
            # TrOCR doesn't expose token-level confidence easily
            # Using a proxy: length-based heuristic
            confidence = min(1.0, len(cleaned) / max(len(raw_text), 1))

            return OCRResult(filename=filename, raw_text=raw_text,
                             cleaned_text=cleaned, confidence=confidence)
        except Exception as e:
            logger.error("TrOCR failed on %s: %s", filename, e)
            return OCRResult(filename=filename, raw_text="", cleaned_text="",
                             confidence=0.0, error=str(e))

    def extract_batch(self, image_list: list, batch_size: int = 8) -> list:
        """
        Batch inference — more efficient than calling extract() in a loop.
        Args:
            image_list : list of (filename, pil_image) tuples
            batch_size : number of images per forward pass
        """
        if not self._loaded:
            self.load()
        import torch
        results = []
        for i in range(0, len(image_list), batch_size):
            batch = image_list[i:i + batch_size]
            filenames = [b[0] for b in batch]
            images    = []
            for _, img in batch:
                pil = self._to_pil(img)
                images.append(pil.convert("RGB") if pil.mode != "RGB" else pil)
            try:
                pixel_values = self.processor(
                    images=images, return_tensors="pt"
                ).pixel_values.to(self.device)
                with torch.no_grad():
                    generated_ids = self.model.generate(pixel_values)
                texts = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
                for fname, raw in zip(filenames, texts):
                    cleaned = self._clean_text(raw.strip())
                    results.append(OCRResult(filename=fname, raw_text=raw,
                                             cleaned_text=cleaned, confidence=0.9))
            except Exception as e:
                logger.error("Batch failed at index %d: %s", i, e)
                for fname, _ in batch:
                    results.append(OCRResult(filename=fname, raw_text="",
                                             cleaned_text="", confidence=0.0, error=str(e)))
        return results

    # ── HELPERS ─────────────────────────────────────────────────────────
    def _to_pil(self, image_input):
        from PIL import Image as PILImage
        if isinstance(image_input, PILImage.Image):
            return image_input
        if isinstance(image_input, np.ndarray):
            return PILImage.fromarray(image_input)
        if isinstance(image_input, (str, Path)):
            return PILImage.open(str(image_input))
        raise TypeError(f"Unsupported type: {type(image_input)}")

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device

    @staticmethod
    def _clean_text(text: str) -> str:
        import re
        return re.sub(r"[^A-Za-z0-9\s]", "", text).strip().upper()


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: python trocr_engine.py <image_path>")
        sys.exit(1)
    engine = TrOCREngine(model_size="base")
    result = engine.extract(sys.argv[1], filename=Path(sys.argv[1]).name)
    print(f"Extracted : '{result.cleaned_text}'")
    print(f"Confidence: {result.confidence:.2f}")
