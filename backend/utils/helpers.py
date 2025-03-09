import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import numpy as np

def preprocess_image_for_classifier(image, size = (224, 224)):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image.convert("RGB")).unsqueeze(0)


def preprocess_image_for_enhancement(image):
    return transforms.ToTensor()(image.convert("RGB")).unsqueeze(0)


def preprocess_image_for_segment(image):
    """Loads and preprocesses an image for the model."""
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    if isinstance(image, Image.Image):  # If the image is a PIL Image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        ])
        image = transform(image)
    
    elif isinstance(image, torch.Tensor):  # If the image is already a tensor
        image = image.unsqueeze(0) if image.ndimension() == 3 else image

    return image

def save_image_as_png(image_tensor, filename, output_dir):
    # Remove batch dimension (assuming batch size is 1)
    image_tensor = image_tensor.squeeze(0)  # shape: [3, 224, 224]
    
    # Convert tensor to numpy array
    image_np = image_tensor.cpu().numpy()  # shape: [3, 224, 224]

    # Transpose the dimensions to [H, W, C]
    image_np = np.transpose(image_np, (1, 2, 0))  # shape: [224, 224, 3]

    # Scale the values to [0, 255] and convert to uint8
    image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)

    # Convert to PIL Image and save
    image_pil = Image.fromarray(image_np)
    image_pil.save(f"{output_dir}/{filename}", format="PNG")

    # print(f"Image saved as {filename}.png")

def save_image(image, filename, path):
    # If the image is a PyTorch tensor, convert to NumPy array
    if isinstance(image, torch.Tensor):
        image = image.cpu().detach().numpy()

    # If the image is a NumPy array, process it to ensure it's in the correct format
    if isinstance(image, np.ndarray):
        # Clip the values to the range 0-255 and convert to uint8
        image = np.clip(image, 0, 255).astype(np.uint8)

        # If the image is grayscale (2D array), convert it to (height, width, 1)
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)  # Convert to (height, width, 1)

        # If the image has only one channel (grayscale with 1 channel), convert to RGB
        if image.ndim == 3 and image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)  # Convert to RGB (height, width, 3)

        # Convert the NumPy array to a PIL Image
        image = Image.fromarray(image)

    # Now that we have a PIL Image, we can save it
    image.save(f"{path}/{filename}", format="PNG")
    # print(f"Saved as {filename}.png")

def conver_mask_numpy(image):
    if isinstance(image, torch.Tensor):
        image = image.squeeze(0)  # Remove batch dimension
        image = image.permute(1, 2, 0).cpu().numpy()  # Move to CPU and convert to NumPy
        image = (image * 255).astype(np.uint8)  # Convert to uint8

    return image

def convert_img_numpy(masked_image):
    if isinstance(masked_image, torch.Tensor):
        masked_image = masked_image.cpu().numpy()  # Move to CPU and convert to NumPy
        masked_image = (masked_image * 255).astype(np.uint8)  # Convert to uint8
    return masked_image


