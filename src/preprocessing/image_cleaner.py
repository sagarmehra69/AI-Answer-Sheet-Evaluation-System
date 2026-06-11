import cv2


class ImageCleaner:
    def __init__(self):
        pass

    def clean_image(self, image_path, output_path):

        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")

        # Upscale
        image = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

        # Gray
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Noise removal
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

        # Adaptive threshold
        processed = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        cv2.imwrite(output_path, processed)

        return output_path
