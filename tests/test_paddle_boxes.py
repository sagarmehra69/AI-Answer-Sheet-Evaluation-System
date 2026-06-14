import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, ROOT_DIR)

from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang="en")

results = ocr.ocr("data/processed/processed_student_001.jpg")

for line in results[0]:
    print(line)
