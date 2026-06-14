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

            # Detect question numbers:
            # Q1, Q2, Q3, Q4
            # Ql, QI (OCR mistakes)
            # Q 1, Q.1, Q-1

            match = re.match(r"^Q[\s\.\-]*([1-9lI])", line, re.IGNORECASE)

            if match:
                question_no = match.group(1)

                if question_no in ["l", "I"]:
                    question_no = "1"

                current_question = f"Q{question_no}"

                # Remove question identifier from line
                answer_text = line[match.end() :].strip()

                questions[current_question] = answer_text

            elif current_question:
                questions[current_question] += " " + line

        # Clean extra spaces
        for qid in questions:
            questions[qid] = " ".join(questions[qid].split())

        return questions
