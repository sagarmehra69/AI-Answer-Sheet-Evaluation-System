"""
MODULE 2 — OCR Ensemble / Manager Layer
Combines multiple OCR engines using confidence-based voting.
Automatically falls back if an engine is not installed.

Your dataset:  FILENAME (e.g. TEST_0001.jpg)  →  IDENTITY (e.g. KEVIN)
This file reads your CSV, runs OCR on each image, compares with ground truth.
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict

# ── Fix import path (works from any directory) ──────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.preprocessing.image_cleaner import ImageCleaner
from src.ocr.tesseract_engine import TesseractEngine
from src.ocr.tesseract_engine import OCRResult as TesseractResult

logger = logging.getLogger(__name__)


# ── Lazy imports for optional engines ───────────────────────────────────────
def _load_trocr():
    try:
        from src.ocr.trocr_engine import TrOCREngine
        return TrOCREngine(model_size="base")
    except Exception as e:
        logger.warning("TrOCR unavailable: %s", e)
        return None

def _load_paddle():
    try:
        from src.ocr.paddle_engine import PaddleOCREngine
        return PaddleOCREngine()
    except Exception as e:
        logger.warning("PaddleOCR unavailable: %s", e)
        return None


# ── Data classes ─────────────────────────────────────────────────────────────
@dataclass
class EnsembleResult:
    filename        : str
    ground_truth    : str = ""          # IDENTITY from CSV
    tesseract_text  : str = ""
    tesseract_conf  : float = 0.0
    trocr_text      : str = ""
    trocr_conf      : float = 0.0
    paddle_text     : str = ""
    paddle_conf     : float = 0.0
    final_text      : str = ""          # Winner after voting
    final_confidence: float = 0.0
    strategy_used   : str = ""          # which engine won
    is_correct      : Optional[bool] = None   # vs ground truth
    cer             : float = 0.0       # character error rate

    def to_dict(self) -> dict:
        return asdict(self)


# ── Main Ensemble Class ──────────────────────────────────────────────────────
class OCREnsemble:
    """
    Orchestrates multiple OCR engines.
    Strategy options:
        'highest_confidence' — pick result from engine with highest confidence
        'majority_vote'      — pick most common text across engines
        'weighted_average'   — confidence-weighted text selection
    """

    def __init__(self,
                 use_tesseract : bool = True,
                 use_trocr     : bool = False,   # set True when installed
                 use_paddle    : bool = False,   # set True when installed
                 strategy      : str  = "highest_confidence",
                 preprocess    : bool = True):
        """
        Args:
            use_tesseract : always True (installed)
            use_trocr     : enable when transformers+torch installed
            use_paddle    : enable when paddleocr installed
            strategy      : voting strategy
            preprocess    : run ImageCleaner before OCR
        """
        self.strategy   = strategy
        self.preprocess = preprocess
        self.cleaner    = ImageCleaner() if preprocess else None

        # Load engines
        self.engines: Dict[str, object] = {}
        if use_tesseract:
            self.engines["tesseract"] = TesseractEngine()
            logger.info("Engine loaded: Tesseract ✅")
        if use_trocr:
            eng = _load_trocr()
            if eng:
                self.engines["trocr"] = eng
                logger.info("Engine loaded: TrOCR ✅")
        if use_paddle:
            eng = _load_paddle()
            if eng:
                self.engines["paddle"] = eng
                logger.info("Engine loaded: PaddleOCR ✅")

        if not self.engines:
            raise RuntimeError("No OCR engines loaded. Install at least pytesseract.")

    # ── SINGLE IMAGE ────────────────────────────────────────────────────
    def process(self, image_input, filename: str = "unknown",
                ground_truth: str = "") -> EnsembleResult:
        """
        Run all engines on one image, combine results.

        Args:
            image_input  : path | numpy array | PIL Image
            filename     : image filename for tracking
            ground_truth : expected text (IDENTITY from CSV)
        """
        # Step 1: Preprocess
        if self.preprocess and self.cleaner:
            try:
                image_input = self.cleaner.clean(image_input)
            except Exception as e:
                logger.warning("Preprocessing failed for %s: %s", filename, e)

        # Step 2: Run all engines
        raw_results = {}
        for name, engine in self.engines.items():
            raw_results[name] = engine.extract(image_input, filename=filename)

        # Step 3: Combine
        result = self._combine(raw_results, filename, ground_truth)
        return result

    # ── BATCH PROCESSING (your CSV dataset) ─────────────────────────────
    def process_csv_dataset(self,
                             csv_path   : str,
                             image_dir  : str,
                             output_json: Optional[str] = None,
                             max_samples: Optional[int] = None) -> pd.DataFrame:
        """
        Process your full dataset from CSV.

        Args:
            csv_path    : path to written_name_train_v2.csv (FILENAME, IDENTITY)
            image_dir   : path to train_v2/ folder (images)
            output_json : if set, save results to this JSON file
            max_samples : limit rows (useful for quick testing)

        Returns:
            DataFrame with OCR results + accuracy metrics
        """
        df = pd.read_csv(csv_path)
        if max_samples:
            df = df.head(max_samples)

        logger.info("Processing %d images from %s", len(df), csv_path)

        results = []
        for idx, row in df.iterrows():
            filename     = row["FILENAME"]
            ground_truth = str(row["IDENTITY"]).strip().upper()
            img_path     = Path(image_dir) / filename

            if not img_path.exists():
                logger.warning("Image not found: %s", img_path)
                continue

            result = self.process(str(img_path), filename=filename,
                                  ground_truth=ground_truth)
            results.append(result.to_dict())

            if (idx + 1) % 50 == 0:
                logger.info("Processed %d/%d", idx + 1, len(df))

        results_df = pd.DataFrame(results)

        # Overall accuracy metrics
        if "is_correct" in results_df.columns:
            acc = results_df["is_correct"].mean()
            avg_cer = results_df["cer"].mean()
            logger.info("=" * 50)
            logger.info("DATASET RESULTS")
            logger.info("  Total images      : %d", len(results_df))
            logger.info("  Exact match acc   : %.2f%%", acc * 100)
            logger.info("  Avg CER           : %.4f", avg_cer)
            logger.info("=" * 50)

        if output_json:
            Path(output_json).parent.mkdir(parents=True, exist_ok=True)
            results_df.to_json(output_json, orient="records", indent=2)
            logger.info("Results saved → %s", output_json)

        return results_df

    # ── COMBINE STRATEGIES ───────────────────────────────────────────────
    def _combine(self, raw_results: dict, filename: str,
                 ground_truth: str) -> EnsembleResult:
        r = EnsembleResult(filename=filename, ground_truth=ground_truth)

        # Fill per-engine results
        if "tesseract" in raw_results:
            r.tesseract_text = raw_results["tesseract"].cleaned_text
            r.tesseract_conf = raw_results["tesseract"].confidence
        if "trocr" in raw_results:
            r.trocr_text = raw_results["trocr"].cleaned_text
            r.trocr_conf = raw_results["trocr"].confidence
        if "paddle" in raw_results:
            r.paddle_text = raw_results["paddle"].cleaned_text
            r.paddle_conf = raw_results["paddle"].confidence

        # Apply strategy
        if self.strategy == "highest_confidence":
            r.final_text, r.final_confidence, r.strategy_used = \
                self._highest_confidence(raw_results)
        elif self.strategy == "majority_vote":
            r.final_text, r.final_confidence, r.strategy_used = \
                self._majority_vote(raw_results)
        else:
            r.final_text, r.final_confidence, r.strategy_used = \
                self._highest_confidence(raw_results)

        # Evaluate vs ground truth
        if ground_truth:
            r.is_correct = (r.final_text.strip() == ground_truth.strip())
            r.cer        = self._char_error_rate(ground_truth, r.final_text)

        return r

    def _highest_confidence(self, raw_results: dict):
        best_name = max(raw_results, key=lambda k: raw_results[k].confidence)
        best      = raw_results[best_name]
        return best.cleaned_text, best.confidence, f"highest_conf:{best_name}"

    def _majority_vote(self, raw_results: dict):
        from collections import Counter
        texts = [v.cleaned_text for v in raw_results.values() if v.cleaned_text]
        if not texts:
            return "", 0.0, "majority_vote:no_text"
        winner = Counter(texts).most_common(1)[0][0]
        # confidence = avg of engines that agreed
        agreeing = [v.confidence for v in raw_results.values()
                    if v.cleaned_text == winner]
        return winner, float(np.mean(agreeing)), "majority_vote"

    @staticmethod
    def _char_error_rate(reference: str, hypothesis: str) -> float:
        """
        Levenshtein-based Character Error Rate.
        CER = edit_distance / len(reference)
        Lower is better. 0.0 = perfect match.
        """
        r, h = list(reference), list(hypothesis)
        d = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
        for i in range(len(r) + 1): d[i][0] = i
        for j in range(len(h) + 1): d[0][j] = j
        for i in range(1, len(r) + 1):
            for j in range(1, len(h) + 1):
                cost = 0 if r[i-1] == h[j-1] else 1
                d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost)
        return d[len(r)][len(h)] / max(len(r), 1)


# ── CLI entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="AASES OCR Ensemble")
    parser.add_argument("--csv",      required=True, help="Path to CSV (FILENAME, IDENTITY)")
    parser.add_argument("--images",   required=True, help="Path to image folder")
    parser.add_argument("--output",   default="data/processed/ocr_results.json")
    parser.add_argument("--samples",  type=int, default=None, help="Limit rows for testing")
    parser.add_argument("--trocr",    action="store_true", help="Enable TrOCR engine")
    parser.add_argument("--paddle",   action="store_true", help="Enable PaddleOCR engine")
    args = parser.parse_args()

    ensemble = OCREnsemble(
        use_tesseract = True,
        use_trocr     = args.trocr,
        use_paddle    = args.paddle,
        strategy      = "highest_confidence"
    )

    results_df = ensemble.process_csv_dataset(
        csv_path    = args.csv,
        image_dir   = args.images,
        output_json = args.output,
        max_samples = args.samples
    )

    print("\n── SUMMARY ─────────────────────────────────────")
    
    # ✅ Fixed
    if len(results_df) == 0:
      print("  ⚠️  No images were processed. Check your paths below.")
    elif "is_correct" in results_df.columns and results_df["is_correct"].notna().any():
      print(f"  Exact match acc  : {results_df['is_correct'].mean()*100:.2f}%")
      print(f"  Avg CER          : {results_df['cer'].mean():.4f}")
    else:
      print("  ⚠️  No ground truth comparisons made.")
    
      print(f"  Images processed : {len(results_df)}")
      print(f"  Results saved    : {args.output}")
