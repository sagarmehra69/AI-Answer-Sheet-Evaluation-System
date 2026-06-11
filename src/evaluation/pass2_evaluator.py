from src.evaluation.semantic_scorer import SemanticScorer
from src.evaluation.rubric_scorer import RubricScorer


class Pass2Evaluator:
    def __init__(self):

        self.semantic_scorer = SemanticScorer()
        self.rubric_scorer = RubricScorer()

    def evaluate(self, student_answer, answer_data):

        model_answer = answer_data["model_answer"]

        rubric_points = answer_data["keywords"]

        max_marks = answer_data["max_marks"]

        # Semantic Score
        semantic_similarity = self.semantic_scorer.calculate_similarity(
            student_answer, model_answer
        )

        # Rubric Score
        rubric_result = self.rubric_scorer.calculate_rubric_score(
            student_answer, rubric_points
        )

        rubric_score = rubric_result["coverage_score"]

        # Pass-2 Hybrid Formula
        final_score = semantic_similarity * 0.6 + rubric_score * 0.4

        marks = round(final_score * max_marks, 2)

        return {
            "semantic_similarity": semantic_similarity,
            "rubric_score": rubric_score,
            "matched_points": rubric_result["matched_points"],
            "missing_points": rubric_result["missing_points"],
            "final_score": round(final_score, 2),
            "marks": marks,
            "max_marks": max_marks,
        }
