import numpy as np
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import tensorflow_hub as hub
from backend.utils.helpers import preprocess_image_for_enhancement
from PIL import Image
import tensorflow as tf

class ESRGAN():
    def __init__(self, model_path="https://tfhub.dev/captain-pool/esrgan-tf2/1"):
        self.esrgan_model = hub.load(model_path)

    def enhance_image(self, image):
        image = tf.image.resize(tf.squeeze(image), (128, 128))

        hr_image = preprocess_image_for_enhancement(image)

        
        fake_image = self.esrgan_model(hr_image)
        
        fake_image = np.squeeze(fake_image.numpy()) 
        
        fake_image = np.clip(fake_image, 0, 255).astype(np.uint8)
        
        fake_image = Image.fromarray(fake_image)
        
        return fake_image

