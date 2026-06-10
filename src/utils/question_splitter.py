import re


class QuestionSplitter:
    def split_questions(self, text):

        lines = text.split("\n")

        questions = {}

        current_question = None

        for line in lines:
            line = line.strip()

            if not line:
                continue

            # Detect Q1, Ql, QI etc.
            if re.match(r"^Q[1-5lI]", line):
                normalized = line.replace("l", "1").replace("I", "1")

                current_question = normalized[:2]

                questions[current_question] = ""

            elif current_question:
                questions[current_question] += " " + line

        return questions
