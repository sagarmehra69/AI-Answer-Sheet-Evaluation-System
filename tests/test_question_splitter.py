import sys
import os

sys.path.append(os.path.abspath("."))

from src.utils.question_splitter import QuestionSplitter

text = """
Q1 What is Artificial Intelligence?
Artificial Intelligence is a branch of computer science.

Q2 What is Machine Learning?
Machine Learning is a subset of AI.

Q3 What is OCR?
OCR converts images into text.
"""

splitter = QuestionSplitter()

questions = splitter.split_questions(text)

print(questions)