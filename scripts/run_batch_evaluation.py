import sys
import os

sys.path.append(os.path.abspath("."))

from src.preprocessing.image_cleaner import ImageCleaner
from src.ocr.paddle_engine import PaddleOCREngine
from src.utils.question_splitter import QuestionSplitter
from src.evaluation.answer_key_manager import AnswerKeyManager
from src.evaluation.pass1_evaluator import Pass1Evaluator


# -----------------------------
# PATHS
# -----------------------------

RAW_IMAGE = "data/raw/student_001.jpg"

PROCESSED_IMAGE = "data/processed/processed_student_001.jpg"

ANSWER_KEY_PATH = "data/sample/answer_key.json"


# -----------------------------
# INITIALIZE MODULES
# -----------------------------

cleaner = ImageCleaner()

ocr_engine = PaddleOCREngine()

splitter = QuestionSplitter()

answer_manager = AnswerKeyManager(ANSWER_KEY_PATH)

evaluator = Pass1Evaluator()


# -----------------------------
# STEP 1: PREPROCESS IMAGE
# -----------------------------

cleaner.clean_image(RAW_IMAGE, PROCESSED_IMAGE)


# -----------------------------
# STEP 2: OCR EXTRACTION
# -----------------------------

ocr_text = ocr_engine.extract_text(PROCESSED_IMAGE)

print("\n===== OCR TEXT =====\n")

print(ocr_text)


# -----------------------------
# STEP 3: QUESTION SPLITTING
# -----------------------------

questions = splitter.split_questions(ocr_text)

print("\n===== DETECTED QUESTIONS =====\n")

print(questions)


# -----------------------------
# STEP 4: EVALUATION
# -----------------------------

print("\n===== FINAL EVALUATION =====\n")

for question_id, student_answer in questions.items():
    answer_data = answer_manager.get_answer(question_id)

    if not answer_data:
        print(f"{question_id} -> No answer key found")

        continue

    result = evaluator.evaluate(student_answer, answer_data)

    print(f"\n{question_id}")

    print(result)
