# src/utils/constants.py

import os

# ─── Project Root ────────────────────────────────────────────
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# ─── Data Paths ──────────────────────────────────────────────
DATA_DIR        = os.path.join(ROOT_DIR, "data")
RAW_DIR         = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR   = os.path.join(DATA_DIR, "processed")
EXPORTS_DIR     = os.path.join(DATA_DIR, "exports")

TRAIN_IMG_DIR   = os.path.join(RAW_DIR, "train_v2")
TEST_IMG_DIR    = os.path.join(RAW_DIR, "test_v2")
VAL_IMG_DIR     = os.path.join(RAW_DIR, "validation_v2")

TRAIN_CSV       = os.path.join(RAW_DIR, "written_name_train_v2.csv")
TEST_CSV        = os.path.join(RAW_DIR, "written_name_test_v2.csv")
VAL_CSV         = os.path.join(RAW_DIR, "written_name_validation_v2.csv")

# ─── Model Paths ─────────────────────────────────────────────
MODELS_DIR      = os.path.join(ROOT_DIR, "models")
OCR_MODEL_DIR   = os.path.join(MODELS_DIR, "ocr")

# ─── Preprocessing Config ────────────────────────────────────
IMAGE_SIZE          = (256, 64)   # (width, height) for OCR input
BINARIZE_THRESHOLD  = 128
DESKEW_MAX_ANGLE    = 10          # degrees
PADDING             = 10          # px padding around cropped text

# ─── OCR Config ──────────────────────────────────────────────
TESSERACT_CONFIG    = "--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz "
CONFIDENCE_THRESHOLD = 60         # minimum Tesseract confidence to trust result
ENSEMBLE_WEIGHTS    = {           # weights for future ensemble voting
    "tesseract": 1.0,
    "paddle":    0.0,             # disabled until installed
    "trocr":     0.0,             # disabled until installed
}

# ─── CSV Columns ─────────────────────────────────────────────
COL_FILENAME    = "FILENAME"
COL_IDENTITY    = "IDENTITY"

# ─── Evaluation Config ───────────────────────────────────────
DISCREPANCY_THRESHOLD = 0.15      # if Pass1 vs Pass2 differ > 15%, flag it

# ─── Logging ─────────────────────────────────────────────────
LOG_LEVEL   = "INFO"
LOG_DIR     = os.path.join(ROOT_DIR, "logs")
LOG_FILE    = os.path.join(LOG_DIR, "aases.log")
