from fastapi import APIRouter

from src.api.schemas import EvaluationRequest

from src.evaluation.answer_key_manager import AnswerKeyManager
from src.evaluation.pass1_evaluator import Pass1Evaluator
from src.evaluation.pass2_evaluator import Pass2Evaluator
from src.evaluation.conflict_resolver import ConflictResolver

router = APIRouter()

answer_manager = AnswerKeyManager("data/sample/answer_key.json")

pass1_evaluator = Pass1Evaluator()
pass2_evaluator = Pass2Evaluator()
conflict_resolver = ConflictResolver()


@router.post("/evaluate")
def evaluate(request: EvaluationRequest):

    answer_data = answer_manager.get_answer(request.question_id)

    pass1_result = pass1_evaluator.evaluate(request.student_answer, answer_data)

    pass2_result = pass2_evaluator.evaluate(request.student_answer, answer_data)

    final_result = conflict_resolver.resolve(pass1_result, pass2_result)

    return {
        "question": request.question_id,
        "pass1": pass1_result,
        "pass2": pass2_result,
        "final": final_result,
    }
