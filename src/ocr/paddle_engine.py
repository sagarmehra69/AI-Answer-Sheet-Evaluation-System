from paddleocr import PaddleOCR


class PaddleOCREngine:
    def __init__(self):

       self.ocr = PaddleOCR(
           use_angle_cls=True,
           lang="en",
           use_gpu=False,
           show_log=False
    )

    def extract_text(self, image_path):

        # Run OCR
        results = self.ocr.ocr(image_path)

        extracted_text = []

        # Extract detected text
        for line in results[0]:

            text = line[1][0]
            confidence = line[1][1]

            if confidence >= 0.70:
                extracted_text.append(text)

        # Join all text
        final_text = "\n".join(extracted_text)

        return final_text
