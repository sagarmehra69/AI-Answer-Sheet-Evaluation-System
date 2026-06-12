import sys
import os

sys.path.append(os.path.abspath("."))

from src.reports.excel_generator import ExcelGenerator


evaluation_results = {
    "Q1": {
        "pass1_result": {"marks": 4.5},
        "pass2_result": {"marks": 5.0},
        "conflict_result": {"final_marks": 4.75},
    },
    "Q2": {
        "pass1_result": {"marks": 7.0},
        "pass2_result": {"marks": 6.5},
        "conflict_result": {"final_marks": 6.75},
    },
}


generator = ExcelGenerator()

generator.generate_report(evaluation_results, "export_results/test_report.xlsx")

print("Excel report generated successfully")
 