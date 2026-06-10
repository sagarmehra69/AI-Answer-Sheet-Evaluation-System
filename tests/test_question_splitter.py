import sys
import os

sys.path.append(os.path.abspath("."))

from src.ocr.paddle_engine import PaddleOCREngine
from src.utils.question_splitter import QuestionSplitter


ocr_engine = PaddleOCREngine()
splitter = QuestionSplitter()

image_path = "data/processed/processed_student_001.jpg"

# OCR extraction
text = ocr_engine.extract_text(image_path)

print("\n===== RAW OCR TEXT =====\n")
print(text)

print("\n========================\n")

# Split questions
questions = splitter.split_questions(text)

print("\n===== SPLIT QUESTIONS =====\n")

print(questions)
