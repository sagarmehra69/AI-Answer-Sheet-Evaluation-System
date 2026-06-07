"""
MODULE 1 — PDF Handler
Converts PDF answer sheets to individual page images for the OCR pipeline.
"""

import logging
from pathlib import Path
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class PDFHandler:
    """
    Converts PDF files to images (one image per page).
    Uses pdf2image (poppler) if available, falls back to PyMuPDF (fitz).
    """

    def __init__(self, dpi: int = 300, output_format: str = "PNG"):
        """
        Args:
            dpi           : render resolution (300 is standard for OCR)
            output_format : 'PNG' or 'JPEG'
        """
        self.dpi           = dpi
        self.output_format = output_format
        self._backend      = self._detect_backend()
        logger.info("PDFHandler using backend: %s", self._backend)

    def _detect_backend(self) -> str:
        try:
            import pdf2image  # noqa
            return "pdf2image"
        except ImportError:
            pass
        try:
            import fitz  # noqa  (PyMuPDF)
            return "pymupdf"
        except ImportError:
            pass
        logger.warning("No PDF backend found. Install: pip install pdf2image OR pymupdf")
        return "none"

    # ── PUBLIC ──────────────────────────────────────────────────────────
    def pdf_to_images(self, pdf_path: str,
                      output_dir: Optional[str] = None) -> List[np.ndarray]:
        """
        Convert each page of a PDF to a numpy array.

        Args:
            pdf_path   : path to PDF file
            output_dir : if set, also save images to this directory

        Returns:
            List of numpy arrays (one per page, BGR)
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if self._backend == "pdf2image":
            images = self._convert_pdf2image(pdf_path)
        elif self._backend == "pymupdf":
            images = self._convert_pymupdf(pdf_path)
        else:
            raise RuntimeError("No PDF backend installed. Run: pip install pdf2image")

        if output_dir:
            self._save_images(images, pdf_path.stem, output_dir)

        logger.info("Converted %d pages from %s", len(images), pdf_path.name)
        return images

    def get_page_count(self, pdf_path: str) -> int:
        """Return number of pages without converting."""
        if self._backend == "pymupdf":
            import fitz
            with fitz.open(str(pdf_path)) as doc:
                return doc.page_count
        if self._backend == "pdf2image":
            from pdf2image import pdfinfo_from_path
            info = pdfinfo_from_path(str(pdf_path))
            return info["Pages"]
        return 0

    # ── BACKENDS ────────────────────────────────────────────────────────
    def _convert_pdf2image(self, pdf_path: Path) -> List[np.ndarray]:
        from pdf2image import convert_from_path
        import cv2
        pil_images = convert_from_path(str(pdf_path), dpi=self.dpi)
        return [cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR) for p in pil_images]

    def _convert_pymupdf(self, pdf_path: Path) -> List[np.ndarray]:
        import fitz, cv2
        images = []
        with fitz.open(str(pdf_path)) as doc:
            for page in doc:
                mat   = fitz.Matrix(self.dpi / 72, self.dpi / 72)
                pix   = page.get_pixmap(matrix=mat)
                arr   = np.frombuffer(pix.samples, dtype=np.uint8)
                arr   = arr.reshape(pix.height, pix.width, pix.n)
                if pix.n == 4:
                    arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
                else:
                    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                images.append(arr)
        return images

    def _save_images(self, images: List[np.ndarray],
                     stem: str, output_dir: str):
        import cv2
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(images):
            path = out / f"{stem}_page_{i + 1:03d}.{self.output_format.lower()}"
            cv2.imwrite(str(path), img)
            logger.debug("Saved page image → %s", path)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG)
    if len(sys.argv) < 2:
        print("Usage: python pdf_handler.py <file.pdf> [output_dir]")
        sys.exit(1)
    handler   = PDFHandler(dpi=300)
    out_dir   = sys.argv[2] if len(sys.argv) > 2 else "pdf_output"
    images    = handler.pdf_to_images(sys.argv[1], output_dir=out_dir)
    print(f"Converted {len(images)} pages → saved to {out_dir}/")
