import cv2


class ImageCleaner:
    def __init__(self):
        pass

    def clean_image(self, image_path, output_path):

        # Read image
        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Denoise image
        denoised = cv2.fastNlMeansDenoising(gray)

        # Adaptive thresholding
        processed = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Save processed image
        cv2.imwrite(output_path, processed)

        return output_path
