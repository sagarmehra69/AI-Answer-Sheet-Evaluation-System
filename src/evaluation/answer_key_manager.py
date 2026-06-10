import json


class AnswerKeyManager:
    def __init__(self, answer_key_path):

        self.answer_key_path = answer_key_path
        self.answer_keys = self.load_answer_keys()

    def load_answer_keys(self):

        with open(self.answer_key_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        return data

    def get_answer(self, question_id):

        return self.answer_keys.get(question_id, {})
