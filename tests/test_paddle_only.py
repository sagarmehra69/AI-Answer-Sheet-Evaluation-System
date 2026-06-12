import sys
import os

sys.path.append(os.path.abspath("."))

from src.ocr.paddle_engine import PaddleOCREngine

engine = PaddleOCREngine()

text = engine.extract_text("data/processed/processed_student_001.jpg")

print(text)
