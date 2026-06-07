"""
MODULE 1 — Page Segmenter
For handwritten name images: detects word/text region bounding boxes.
For full answer sheets: segments individual question-answer blocks.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TextRegion:
    """Represents a detected text region in an image."""
    region_id  : int
    x          : int
    y          : int
    width      : int
    height     : int
    confidence : float = 1.0
    label      : str   = ""          # e.g. "question_1", "name_field"

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)

    @property
    def crop_coords(self) -> Tuple[int, int, int, int]:
        """Returns (x1, y1, x2, y2) for numpy slicing."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)


class PageSegmenter:
    """
    Detects and segments text regions from preprocessed images.

    Two modes:
        'word'  — single word/name image (IAM-style handwriting dataset)
        'page'  — full answer sheet with multiple question blocks
    """

    def __init__(self, mode: str = "word", min_area: int = 100):
        """
        Args:
            mode     : 'word' for single-word images, 'page' for answer sheets
            min_area : minimum pixel area to count as a text region
        """
        assert mode in ("word", "page"), "mode must be 'word' or 'page'"
        self.mode     = mode
        self.min_area = min_area

    # ── PUBLIC ──────────────────────────────────────────────────────────
    def segment(self, img: np.ndarray) -> List[TextRegion]:
        """
        Main method — detect text regions in image.

        Args:
            img : grayscale numpy array (already cleaned by ImageCleaner)

        Returns:
            List of TextRegion objects sorted top-to-bottom, left-to-right
        """
        if self.mode == "word":
            return self._segment_word(img)
        return self._segment_page(img)

    def crop_regions(self, img: np.ndarray,
                     regions: List[TextRegion]) -> List[Tuple[TextRegion, np.ndarray]]:
        """
        Crop detected regions from image.

        Returns:
            List of (TextRegion, cropped_numpy_array)
        """
        crops = []
        for region in regions:
            x1, y1, x2, y2 = region.crop_coords
            crop = img[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append((region, crop))
        return crops

    def draw_regions(self, img: np.ndarray,
                     regions: List[TextRegion]) -> np.ndarray:
        """Draw bounding boxes on image for visualization/debugging."""
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()
        for r in regions:
            cv2.rectangle(vis, (r.x, r.y), (r.x + r.width, r.y + r.height),
                          (0, 255, 0), 2)
            cv2.putText(vis, f"R{r.region_id}", (r.x, r.y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return vis

    # ── SEGMENTATION MODES ──────────────────────────────────────────────
    def _segment_word(self, img: np.ndarray) -> List[TextRegion]:
        """
        For single handwritten word images (your dataset).
        Finds the main text bounding box — returns as single region.
        """
        inverted = cv2.bitwise_not(img)
        coords   = cv2.findNonZero(inverted)
        if coords is None:
            logger.warning("No text pixels found in image")
            return []
        x, y, w, h = cv2.boundingRect(coords)
        region = TextRegion(region_id=0, x=x, y=y, width=w, height=h,
                            confidence=1.0, label="handwritten_word")
        return [region]

    def _segment_page(self, img: np.ndarray) -> List[TextRegion]:
        """
        For full answer sheet pages.
        Uses morphological operations to find paragraph/answer blocks.
        """
        # Dilate to merge nearby text into blocks
        kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
        dilated = cv2.dilate(cv2.bitwise_not(img), kernel, iterations=3)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        for idx, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area < self.min_area:
                continue
            regions.append(TextRegion(region_id=idx, x=x, y=y,
                                      width=w, height=h, label=f"block_{idx}"))

        # Sort top-to-bottom (reading order)
        regions.sort(key=lambda r: (r.y, r.x))
        logger.debug("Segmented %d regions", len(regions))
        return regions


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG)
    if len(sys.argv) < 2:
        print("Usage: python page_segmenter.py <cleaned_image.png>")
        sys.exit(1)
    img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    seg = PageSegmenter(mode="word")
    regions = seg.segment(img)
    print(f"Found {len(regions)} region(s)")
    for r in regions:
        print(f"  Region {r.region_id}: bbox={r.bbox} label={r.label}")
    vis = seg.draw_regions(img, regions)
    cv2.imwrite("segmented_output.png", vis)
    print("Saved → segmented_output.png")
