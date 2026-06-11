class ConflictResolver:
    def __init__(self, threshold=2):

        self.threshold = threshold

    def resolve(self, pass1_result, pass2_result):

        pass1_marks = pass1_result["marks"]
        pass2_marks = pass2_result["marks"]

        difference = abs(pass1_marks - pass2_marks)

        if difference <= self.threshold:
            final_marks = round((pass1_marks + pass2_marks) / 2, 2)

            return {
                "status": "accepted",
                "difference": difference,
                "final_marks": final_marks,
            }

        return {
            "status": "human_review_required",
            "difference": difference,
            "final_marks": None,
        }
