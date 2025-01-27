import os
import sys
import warnings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
torch.cuda.empty_cache()

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from backend.models.baseline_resnet50 import BASELINE_RESNET50
from backend.models.unet import UNET
from backend.models.esrgan import ESRGAN
from backend.utils.helpers import save_image


# baseline_model = BASELINE_RESNET50()
# esrgan_model = ESRGAN()
# unet_model = UNET()

class_labels = [
    'Abudefduf Vaigiensis', 'Acanthurus Nigrofuscus', 'Balistapus Undulatus', 'Canthigaster Valentini', 
    'Chaetodon Trifascialis', 'Hemigymnus Fasciatus', 'Hemigymnus Melapterus', 'Lutjanus fulvus', 
    'Myripristis Kuntee', 'Neoglyphidodon Nigroris', 'Neoniphon Sammara', 'Pempheris Vanicolensis', 
    'Pomacentrus Moluccensis', 'Scaridae', 'Scolopsis Bilineata', 'Siganus Fuscescens', 'Zanclus Cornutus', 
    'Zebrasoma Scopas'
]

test_images_dir = 'test/images'
output_dir = 'test/results'

def main():
    # predict one image
    image_path = 'test/images/fish_000065789596_04756.png'
    process_single_image(image_path, output_dir)

    # predict all image in dir
    # process_images_in_directory(test_images_dir, output_dir)


def process_single_image(image_path, output_dir):
    # Load image
    print('Loading image:', image_path)
    image = Image.open(image_path)
    print('Image loaded...')
    print('Enhancing image...')
    enhanced_image = ESRGAN().enhance_image(image)
    print('Image enhanced...')

    print('Masking image...')
    masked_image = UNET().mask(enhanced_image)
    print('Image masked...')
    
    # Save the images for checking
    filename = os.path.basename(image_path)
    print(f"Saving Enhanced image: {filename}")
    save_image(enhanced_image, {"enhanced - " + filename}, output_dir)
    save_image(masked_image, {"masked - " + filename}, output_dir)
    print(f"Image saved: {filename}")
    
    # Classify the image using baseline ResNet50
    print("Classifying the image...")
    predicted_class = BASELINE_RESNET50().predict(masked_image)
    predicted_class_name = class_labels[predicted_class]
    print(f"Prediction: {predicted_class_name}")
    
    return predicted_class_name


def process_images_in_directory(test_images_dir, output_dir):
    results = []
    
    for filename in os.listdir(test_images_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  
            image_path = os.path.join(test_images_dir, filename)
            
            predicted_class_name = process_single_image(image_path, output_dir)
            
            results.append(f"{filename}: {predicted_class_name}")
    
    for result in results:
        print(result)


main()
