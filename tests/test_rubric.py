import sys
import os

sys.path.append(os.path.abspath("."))

from src.evaluation.rubric_scorer import RubricScorer

scorer = RubricScorer()

student_answer = """
Artificial Intelligence enables machines
to learn and make decisions.
"""

rubric_points = [
    "Artificial Intelligence",
    "human intelligence",
    "learning",
    "decision making"
]

result = scorer.calculate_rubric_score(
    student_answer,
    rubric_points
)

print(result)