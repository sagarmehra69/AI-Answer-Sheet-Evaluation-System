import sys
import os

# ----------------------------------
# PROJECT ROOT
# ----------------------------------

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, ROOT_DIR)

# ----------------------------------
# EVALUATION IMPORTS FIRST
# (Avoid Paddle/Torch DLL conflict)
# ----------------------------------

from src.evaluation.answer_key_manager import AnswerKeyManager
from src.evaluation.pass1_evaluator import Pass1Evaluator
from src.evaluation.pass2_evaluator import Pass2Evaluator
from src.evaluation.conflict_resolver import ConflictResolver
from src.reports.excel_generator import ExcelGenerator
from src.reports.pdf_generator import PDFGenerator

# ----------------------------------
# OCR IMPORTS SECOND
# ----------------------------------

from src.preprocessing.image_cleaner import ImageCleaner
from src.ocr.paddle_engine import PaddleOCREngine
from src.utils.question_splitter import QuestionSplitter

# ----------------------------------
# PATHS
# ----------------------------------

RAW_IMAGE = "data/raw/student_001.jpg"

PROCESSED_IMAGE = "data/processed/processed_student_001.jpg"

ANSWER_KEY_PATH = "data/sample/answer_key.json"

# ----------------------------------
# LOAD ANSWER KEY + EVALUATOR FIRST
# ----------------------------------

answer_manager = AnswerKeyManager(ANSWER_KEY_PATH)

pass1_evaluator = Pass1Evaluator()
pass2_evaluator = Pass2Evaluator()
conflict_resolver = ConflictResolver()

# ----------------------------------
# OCR COMPONENTS
# ----------------------------------

cleaner = ImageCleaner()

ocr_engine = PaddleOCREngine()

splitter = QuestionSplitter()

# ----------------------------------
# STEP 1: PREPROCESS IMAGE
# ----------------------------------

print("\n===== IMAGE PREPROCESSING =====\n")

cleaner.clean_image(RAW_IMAGE, PROCESSED_IMAGE)

print("Image processed successfully")

# ----------------------------------
# STEP 2: OCR
# ----------------------------------

print("\n===== OCR EXTRACTION =====\n")

ocr_text = ocr_engine.extract_text(PROCESSED_IMAGE)

print(ocr_text)

# ----------------------------------
# STEP 3: QUESTION SPLITTING
# ----------------------------------

print("\n===== DETECTED QUESTIONS =====\n")

questions = splitter.split_questions(ocr_text)

print(questions)

# ----------------------------------
# STEP 4: EVALUATION
# ----------------------------------

print("\n===== FINAL EVALUATION =====\n")

evaluation_results = {}

for question_id, student_answer in questions.items():
    answer_data = answer_manager.get_answer(question_id)

    if not answer_data:
        print(f"{question_id} -> No answer key found")
        continue

    # -------------------------
    # PASS 1
    # -------------------------

    pass1_result = pass1_evaluator.evaluate(student_answer, answer_data)

    # -------------------------
    # PASS 2
    # -------------------------

    pass2_result = pass2_evaluator.evaluate(student_answer, answer_data)

    # -------------------------
    # CONFLICT RESOLUTION
    # -------------------------

    final_result = conflict_resolver.resolve(pass1_result, pass2_result)

    # -------------------------
    # STORE RESULTS
    # -------------------------

    evaluation_results[question_id] = {
        "pass1_result": pass1_result,
        "pass2_result": pass2_result,
        "conflict_result": final_result,
    }

    # -------------------------
    # OUTPUT
    # -------------------------

    print(f"\n{'=' * 50}")
    print(question_id)
    print(f"{'=' * 50}")

    print("\nPASS 1 RESULT")
    print(pass1_result)

    print("\nPASS 2 RESULT")
    print(pass2_result)

    print("\nCONFLICT RESOLUTION")
    print(final_result)


print("\n===== SAVED RESULTS =====\n")
print(evaluation_results)

excel_generator = ExcelGenerator()
pdf_generator = PDFGenerator()

print("\n===== EVALUATION COMPLETED =====")

excel_generator.generate_report(
    evaluation_results,
    "export_results/evaluation_report.xlsx"
)

pdf_generator.generate_report(
    evaluation_results,
    "export_results/evaluation_report.pdf"
)

print("\n===== REPORTS GENERATED =====")