from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet


class PDFGenerator:
    def generate_report(self, evaluation_results, output_file):

        pdf = SimpleDocTemplate(output_file)

        styles = getSampleStyleSheet()

        elements = []

        title = Paragraph("AI Answer Sheet Evaluation Report", styles["Title"])

        elements.append(title)

        elements.append(Spacer(1, 20))

        data = [
            [
                "Question",
                "Pass1 Marks",
                "Pass2 Marks",
                "Final Marks",
            ]
        ]

        total_marks = 0

        for question_id, result in evaluation_results.items():
            final_marks = result["conflict_result"]["final_marks"]

            total_marks += final_marks

            data.append(
                [
                    question_id,
                    result["pass1_result"]["marks"],
                    result["pass2_result"]["marks"],
                    final_marks,
                ]
            )

        data.append(
            [
                "TOTAL",
                "",
                "",
                round(total_marks, 2),
            ]
        )

        table = Table(data)

        table.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ]
            )
        )

        elements.append(table)

        pdf.build(elements)

        return output_file
