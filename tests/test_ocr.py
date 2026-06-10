import sys
import os

sys.path.append(os.path.abspath("."))

from src.ocr.paddle_engine import PaddleOCREngine


ocr_engine = PaddleOCREngine()

image_path = "data/processed/processed_student_001.jpg"

text = ocr_engine.extract_text(image_path)

print("\n===== EXTRACTED TEXT =====\n")
print(text)
