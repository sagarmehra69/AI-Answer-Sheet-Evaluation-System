from pydantic import BaseModel


class EvaluationRequest(BaseModel):
    question_id: str
    student_answer: str
