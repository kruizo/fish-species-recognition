import numpy as np
import tensorflow_hub as hub
from backend.utils.helpers import preprocess_image_for_enhancement
from PIL import Image
from tensorflow import keras
import tensorflow as tf

class UNET():
    def __init__(self, model_path="backend/models/weights/best_unetmodel.keras"):
        self.unet_model = keras.models.load_model(model_path, compile=False)

    def mask(self, image):
        predicted_mask2 = self.unet_model.predict(np.expand_dims(image, axis=0))

        predicted_mask2 = np.squeeze(predicted_mask2)

        binary_mask2 = (predicted_mask2 > 0.1).astype(np.uint8)

        masked_image_pred2 = image * np.expand_dims(binary_mask2, axis=-1)


        return masked_image_pred2
    
        # image = preprocess_image_for_enhancement(image)
        
        # # Resize the image to the desired dimensions
        # image = tf.image.resize(tf.squeeze(image), (128, 128))
        
        # # Ensure the image is clipped and converted to uint8
        # image = tf.clip_by_value(image, 0, 255).numpy().astype(np.uint8)

        # # Predict the mask using the UNET model
        # predicted_mask = self.unet_model.predict(np.expand_dims(image, axis=0))[0]
        
        # # Squeeze the predicted mask to remove unnecessary dimensions
        # predicted_mask = np.squeeze(predicted_mask)

        # # Apply a binary threshold to create a binary mask
        # binary_mask = (predicted_mask > 0.5).astype(np.uint8)  # Threshold 0.3

        # # Post-process: Apply the mask to the input image
        # masked_image_pred = image * np.expand_dims(binary_mask, axis=-1)

        # # Clip and ensure masked image is in uint8 format
        # masked_image_pred = np.clip(masked_image_pred, 0, 255).astype(np.uint8)
        
        # # Convert the masked image to a PIL Image
        # masked_image_pred = Image.fromarray(masked_image_pred)


