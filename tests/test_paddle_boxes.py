import sys
import os

sys.path.append(os.path.abspath("."))

from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False, show_log=False)

results = ocr.ocr("data/processed/processed_student_001.jpg")

for item in results[0]:
    print(item)
