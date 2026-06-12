import sys
import os

sys.path.append(os.path.abspath("."))

from src.reports.pdf_generator import PDFGenerator


sample_results = {
    "Q1": {
        "pass1_result": {"marks": 5},
        "pass2_result": {"marks": 6},
        "conflict_result": {"final_marks": 5.5},
    },
    "Q2": {
        "pass1_result": {"marks": 7},
        "pass2_result": {"marks": 8},
        "conflict_result": {"final_marks": 7.5},
    },
}

generator = PDFGenerator()

generator.generate_report(
    sample_results,
    "export_results/test_report.pdf",
)

print("PDF Generated Successfully")
