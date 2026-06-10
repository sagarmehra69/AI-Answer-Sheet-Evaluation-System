class KeywordScorer:
    def __init__(self):
        pass

    def calculate_keyword_score(self, student_answer, keywords):

        student_answer = student_answer.lower()

        matched_keywords = []

        for keyword in keywords:
            if keyword.lower() in student_answer:
                matched_keywords.append(keyword)

        total_keywords = len(keywords)

        matched_count = len(matched_keywords)

        if total_keywords == 0:
            return 0, []

        score = matched_count / total_keywords

        return round(score, 2), matched_keywords
