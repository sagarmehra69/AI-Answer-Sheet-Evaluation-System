import sys
import os

sys.path.append(os.path.abspath("."))

from src.ocr.paddle_engine import PaddleOCREngine
from src.ocr.tesseract_engine import TesseractEngine
from src.ocr.trocr_engine import TrOCREngine

IMAGE_PATH = "data/processed/processed_student_001.jpg"

print("\n===== PADDLE OCR =====\n")

paddle = PaddleOCREngine()

print(paddle.extract_text(IMAGE_PATH))

print("\n===== TESSERACT OCR =====\n")

tesseract = TesseractEngine()

print(tesseract.extract_text(IMAGE_PATH))

print("\n===== TrOCR =====\n")

trocr = TrOCREngine()

print(trocr.extract_text(IMAGE_PATH))
