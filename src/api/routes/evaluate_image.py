from fastapi import APIRouter, UploadFile, File
import os

from src.preprocessing.image_cleaner import ImageCleaner
from src.ocr.paddle_engine import PaddleOCREngine
from src.utils.question_splitter import QuestionSplitter

from src.evaluation.answer_key_manager import AnswerKeyManager
from src.evaluation.pass1_evaluator import Pass1Evaluator
from src.evaluation.pass2_evaluator import Pass2Evaluator
from src.evaluation.conflict_resolver import ConflictResolver

from src.database.crud import save_result

router = APIRouter()

cleaner = ImageCleaner()
ocr_engine = PaddleOCREngine()
splitter = QuestionSplitter()

answer_manager = AnswerKeyManager("data/sample/answer_key.json")

pass1_evaluator = Pass1Evaluator()
pass2_evaluator = Pass2Evaluator()
conflict_resolver = ConflictResolver()


@router.post("/evaluate-image")
async def evaluate_image(file: UploadFile = File(...)):

    upload_dir = "data/uploads"
    os.makedirs(upload_dir, exist_ok=True)

    image_path = os.path.join(upload_dir, file.filename)

    with open(image_path, "wb") as buffer:
        buffer.write(await file.read())

    processed_path = f"data/processed/processed_{file.filename}"

    cleaner.clean_image(image_path, processed_path)

    ocr_text = ocr_engine.extract_text(processed_path)

    questions = splitter.split_questions(ocr_text)

    evaluation_results = {}

    for question_id, student_answer in questions.items():
        answer_data = answer_manager.get_answer(question_id)

        if not answer_data:
            continue

        pass1_result = pass1_evaluator.evaluate(student_answer, answer_data)

        pass2_result = pass2_evaluator.evaluate(student_answer, answer_data)

        final_result = conflict_resolver.resolve(pass1_result, pass2_result)

        save_result(
            question_id,
            pass1_result["marks"],
            pass2_result["marks"],
            final_result["final_marks"],
        )

        evaluation_results[question_id] = {
            "pass1": pass1_result,
            "pass2": pass2_result,
            "final": final_result,
        }

    return {"ocr_text": ocr_text, "questions": questions, "results": evaluation_results}
