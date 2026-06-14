import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, ROOT_DIR)

from src.ocr.paddle_engine import PaddleOCREngine

ocr = PaddleOCREngine()

text = ocr.extract_text("data/processed/processed_student_001.jpg")

print("\n===== OCR OUTPUT =====\n")
print(text)
