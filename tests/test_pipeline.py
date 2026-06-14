import sys
import os

sys.path.append(os.path.abspath("."))

from src.utils.question_splitter import QuestionSplitter

sample_text = """
Q1 Artificial Intelligence is a branch of computer science.

Q2 Machine Learning is a subset of AI.

Q3 OCR converts images into text.
"""

splitter = QuestionSplitter()

print(splitter.split_questions(sample_text))
