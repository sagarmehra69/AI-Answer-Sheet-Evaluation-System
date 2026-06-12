import pytesseract


class TesseractEngine:
    def __init__(self):

        pytesseract.pytesseract.tesseract_cmd = (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        )

    def extract_text(self, image_path):

        text = pytesseract.image_to_string(image_path, lang="eng")

        return text
