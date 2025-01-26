import os
import sys
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("Python path:", sys.path)

from backend.models.baseline_resnet50 import BASELINE_RESNET50
from backend.models.esrgan import ESRGAN
from backend.utils.helpers import preprocess_image_for_classifier, save_image
from PIL import Image

def process_single_image(image_path, output_dir):
    # Load image
    image = Image.open(image_path)
    
    # Enhance the image using ESRGAN
    enhanced_image = esrgan_loader.enhance_image(image)
    
    # Save the enhanced image
    filename = os.path.basename(image_path)
    print(f"Saving image: {filename}")
    save_image(enhanced_image, filename, output_dir)
    print(f"Image saved: {filename}")
    
    # Preprocess the image for classification
    processed_image = preprocess_image_for_classifier(enhanced_image)

    # Classify the image using baseline ResNet50
    predicted_class = baseline_loader.predict(processed_image)
    predicted_class_name = class_labels[predicted_class]
    print(f"Prediction: {predicted_class_name}")
    
    return predicted_class_name


def process_images_in_directory(test_images_dir, output_dir):
    results = []
    
    # Iterate through all images in the directory
    for filename in os.listdir(test_images_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # Ensure it's an image file
            image_path = os.path.join(test_images_dir, filename)
            
            # Process the image
            predicted_class_name = process_single_image(image_path, output_dir)
            
            # Store the result
            results.append(f"{filename}: {predicted_class_name}")
    
    # Print all results
    for result in results:
        print(result)


baseline_loader = BASELINE_RESNET50()
esrgan_loader = ESRGAN()

class_labels = [
    'Abudefduf Vaigiensis', 'Acanthurus Nigrofuscus', 'Balistapus Undulatus', 'Canthigaster Valentini', 
    'Chaetodon Trifascialis', 'Hemigymnus Fasciatus', 'Hemigymnus Melapterus', 'Lutjanus fulvus', 
    'Myripristis Kuntee', 'Neoglyphidodon Nigroris', 'Neoniphon Sammara', 'Pempheris Vanicolensis', 
    'Pomacentrus Moluccensis', 'Scaridae', 'Scolopsis Bilineata', 'Siganus Fuscescens', 'Zanclus Cornutus', 
    'Zebrasoma Scopas'
]
test_images_dir = 'test/images'
output_dir = 'test/results'


# Process a single image
image_path = 'test/images/Chaetodon Trifascialis.jpg'
process_single_image(image_path, output_dir)

# Process all images in the directory
process_images_in_directory(test_images_dir, output_dir)
