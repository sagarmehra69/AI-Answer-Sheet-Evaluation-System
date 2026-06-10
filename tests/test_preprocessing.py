import sys
import os

sys.path.append(os.path.abspath("."))

from src.preprocessing.image_cleaner import ImageCleaner


cleaner = ImageCleaner()

input_path = "data/raw/student_001.jpg"
output_path = "data/processed/processed_student_001.jpg"

cleaner.clean_image(input_path, output_path)

print("Preprocessing completed successfully.")
