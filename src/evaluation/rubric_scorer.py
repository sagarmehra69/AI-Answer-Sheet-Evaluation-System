class RubricScorer:
    def __init__(self):
        pass

    def calculate_rubric_score(self, student_answer, rubric_points):

        student_answer = student_answer.lower()

        matched_points = []
        missing_points = []

        for point in rubric_points:
            if point.lower() in student_answer:
                matched_points.append(point)

            else:
                missing_points.append(point)

        total_points = len(rubric_points)

        if total_points == 0:
            return {"coverage_score": 0, "matched_points": [], "missing_points": []}

        coverage_score = len(matched_points) / total_points

        return {
            "coverage_score": round(coverage_score, 2),
            "matched_points": matched_points,
            "missing_points": missing_points,
        }
