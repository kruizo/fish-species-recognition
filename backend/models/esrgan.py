import numpy as np
import tensorflow_hub as hub
from backend.utils.helpers import preprocess_image_for_enhancement
from PIL import Image

class ESRGAN():
    def __init__(self, model_path="https://tfhub.dev/captain-pool/esrgan-tf2/1"):
        self.esrgan_model = hub.load(model_path)

    def enhance_image(self, image):
        # Preprocess the image for ESRGAN enhancement
        hr_image = preprocess_image_for_enhancement(image)
        fake_image = self.esrgan_model(hr_image)
        
        # Remove the batch dimension and any extra dimensions, if present
        fake_image = np.squeeze(fake_image.numpy())  # Remove batch dimension and extra dimensions
        
        # Ensure the pixel values are in the range [0, 255]
        fake_image = np.clip(fake_image, 0, 255).astype(np.uint8)
        
        # Convert the NumPy array to a PIL Image
        fake_image = Image.fromarray(fake_image)
        
        # Return the enhanced PIL image
        return fake_image

