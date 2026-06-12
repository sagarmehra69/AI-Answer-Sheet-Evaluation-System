import sys
import os

sys.path.append(os.path.abspath("."))

from src.ocr.paddle_engine import PaddleOCREngine

ocr = PaddleOCREngine()

text = ocr.extract_text("data/raw/q1_test.jpg")

print("\n===== Q1 OCR =====\n")

print(text)
