import sys
import os

sys.path.append(os.path.abspath("."))

from src.evaluation.answer_key_manager import AnswerKeyManager
from src.evaluation.pass1_evaluator import Pass1Evaluator


# Load answer key
answer_manager = AnswerKeyManager("data/sample/answer_key.json")

# Evaluator
evaluator = Pass1Evaluator()


# Example student answer
student_answer = """
Artificial Intelligence allows machines
to simulate human intelligence and
perform tasks automatically.
"""

# Load answer data
answer_data = answer_manager.get_answer("Q1")


# Evaluate
result = evaluator.evaluate(student_answer, answer_data)

print("\n===== EVALUATION RESULT =====\n")

print(result)
