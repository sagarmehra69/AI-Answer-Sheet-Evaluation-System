from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


class TrOCREngine:
    def __init__(self):

        self.processor = TrOCRProcessor.from_pretrained(
            "microsoft/trocr-base-handwritten"
        )

        self.model = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-base-handwritten"
        )

    def extract_text(self, image_path):

        image = Image.open(image_path).convert("RGB")

        pixel_values = self.processor(image, return_tensors="pt").pixel_values

        generated_ids = self.model.generate(pixel_values)

        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return text
