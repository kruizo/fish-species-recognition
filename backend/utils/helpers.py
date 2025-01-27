import numpy as np
import tensorflow as tf
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import tensorflow as tf

# def preprocess_image_for_classifier(file):
#     image = Image.open(io.BytesIO(file.read()))
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     return transform(image).unsqueeze(0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
def preprocess_image_for_classifier(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    image = image.convert('RGB')

    image = Image.open(image).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

def preprocess_image_for_enhancement(image):
    """Loads and preprocesses an image for the model."""
    
    hr_image = np.array(image)
    if hr_image.shape[-1] == 4:
        hr_image = hr_image[...,:-1]  
    
    hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
    hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
    
    hr_image = tf.cast(hr_image, tf.float32)
    return tf.expand_dims(hr_image, 0)  

def preprocess_image_for_segment(image):
    """Loads and preprocesses an image for the model."""
    
    hr_image = np.array(image)
    
    if hr_image.shape[-1] == 4:
        hr_image = hr_image[...,:-1] 
    
    hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
    hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
    
    hr_image = tf.cast(hr_image, tf.float32)
    return tf.expand_dims(hr_image, 0)  

def save_image(image, filename, path):
    if not isinstance(image, Image.Image):
       
        if image.ndim == 4:  
            image = tf.squeeze(image, axis=0)

        image = tf.clip_by_value(image, 0, 255)  
        
        image = image.numpy()
        
        image = Image.fromarray(image.astype(np.uint8))
    
    image.save(f"{path}/{filename}.jpg")
    print(f"Saved as {filename}.jpg")

def pad_to_square(image):
    # Get original dimensions
    old_h, old_w = image.shape[:2]
    
    # Find the larger dimension to make the image square
    max_dim = max(old_h, old_w)
    
    # Calculate padding
    pad_h = (max_dim - old_h) // 2
    pad_w = (max_dim - old_w) // 2
    
    # Ensure padding is non-negative and compatible
    padded_image = tf.image.pad_to_bounding_box(image, pad_h, pad_w, max_dim, max_dim)
    
    # Handle any excess due to rounding differences
    padded_image = tf.image.resize_with_crop_or_pad(padded_image, max_dim, max_dim)
    
    return padded_image