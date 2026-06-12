from openpyxl import Workbook


class ExcelGenerator:
    def generate_report(self, evaluation_results, output_file):

        workbook = Workbook()

        sheet = workbook.active

        sheet.title = "Evaluation Report"

        # Headers
        sheet.append(
            [
                "Question",
                "Pass1 Marks",
                "Pass2 Marks",
                "Final Marks",
            ]
        )

        total_marks = 0

        for question_id, result in evaluation_results.items():
            final_marks = result["conflict_result"]["final_marks"]

            total_marks += final_marks

            sheet.append(
                [
                    question_id,
                    result["pass1_result"]["marks"],
                    result["pass2_result"]["marks"],
                    final_marks,
                ]
            )

        sheet.append([])

        sheet.append(["TOTAL", "", "", total_marks])

        workbook.save(output_file)

        return output_file
