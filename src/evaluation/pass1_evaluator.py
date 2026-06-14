from src.evaluation.semantic_scorer import SemanticScorer
from src.evaluation.keyword_scorer import KeywordScorer


class Pass1Evaluator:
    def __init__(self):

        self.semantic_scorer = SemanticScorer()
        self.keyword_scorer = KeywordScorer()

    def evaluate(self, student_answer, answer_data):

        model_answer = answer_data["model_answer"]
        keywords = answer_data["keywords"]
        max_marks = answer_data["max_marks"]

        # Semantic score
        semantic_similarity = self.semantic_scorer.calculate_similarity(
            student_answer, model_answer
        )

        # Keyword score
        keyword_score, matched_keywords = self.keyword_scorer.calculate_keyword_score(
            student_answer, keywords
        )

        # Hybrid score
        final_score = (semantic_similarity * 0.85) + (keyword_score * 0.15)

        obtained_marks = round(final_score * max_marks, 2)

        if semantic_similarity >= 0.80:
            final_score = max(final_score, 0.80)

        return {
            "semantic_similarity": semantic_similarity,
            "keyword_score": keyword_score,
            "matched_keywords": matched_keywords,
            "final_score": round(final_score, 2),
            "marks": obtained_marks,
            "max_marks": max_marks,
        }
