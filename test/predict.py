import os
import sys
import warnings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import torch
torch.cuda.empty_cache()

from PIL import Image, ImageFile

from backend.models.baseline import BASELINE_RESNET50
from backend.models.proposed import PROPOSED_RESNET50
from backend.models.unet import UNET
from backend.models.esrgan import ESRGAN
from backend.utils.helpers import convert_img_numpy, conver_mask_numpy, save_image, save_image_as_png


class_labels = [
    'Abudefduf Vaigiensis', 'Acanthurus Nigrofuscus', 'Balistapus Undulatus', 'Canthigaster Valentini', 
    'Chaetodon Trifascialis', 'Hemigymnus Fasciatus', 'Hemigymnus Melapterus', 'Lutjanus fulvus', 
    'Myripristis Kuntee', 'Neoglyphidodon Nigroris', 'Neoniphon Sammara', 'Pempheris Vanicolensis', 
    'Pomacentrus Moluccensis', 'Scaridae', 'Scolopsis Bilineata', 'Siganus Fuscescens', 'Zanclus Cornutus', 
    'Zebrasoma Scopas'
]

test_images_dir = 'test/images'
masked_output_dir = 'test/results/masked'
enhanced_output_dir = 'test/results/enhanced'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def run():
    # predict one image
    # image_path = 'test/images/fish_000026690001_02685.png'
    # process_single_image(image_path)

    # predict all image in dir
    process_images_in_directory(test_images_dir)

def process_single_image(image_path):
    print('Image:', image_path)
    image = Image.open(image_path)
    # print('Image loaded...')

    bpredicted_class, bconfiderence, bprediction_time = BASELINE_RESNET50(device=device).predict(image)

    filename = os.path.basename(image_path)

    # print('Enhancing image...')
    enhanced_image, time_taken = ESRGAN(device=device).predict(image)
    # print('Image enhanced... TIME TAKEN: ', time_taken)
    # print('Image enhanced SHAPE:', enhanced_image.shape)
    # print(f"Saving Enhanced image: {filename}")
    
    save_image_as_png(enhanced_image, "enhanced - " + filename, enhanced_output_dir)
    # print(f"Image saved: {filename}")
    
    # print('Masking image...')
    binary_mask = UNET(device=device).predict(enhanced_image)
    # print('Image masked...')
    
    # print('Mask SHAPE:', binary_mask.shape)

    enhanced_image = enhanced_image.squeeze().cpu().numpy().transpose(1, 2, 0)

    masked_image = enhanced_image * np.expand_dims(binary_mask, axis=-1)
    
    masked_image = np.clip(masked_image * 255, 0, 255).astype(np.uint8)
    
    # print("Image shapes: ", enhanced_image.shape, masked_image.shape, binary_mask.shape)

    # print(f"Saving Masked image: {filename}")
    save_image(masked_image, "masked - " + filename, masked_output_dir)
    # print(f"Image saved: {filename}")
    
    # print("Classifying the image...")
    predicted_class, confidence, prediction_time = PROPOSED_RESNET50(device=device).predict(masked_image)
    print(f"BASELINE Prediction: {class_labels[bpredicted_class]} | Confidence: {bconfiderence:.2f} | Prediction Time: {bprediction_time:.3f}s")
    print(f"PROPOSED Prediction: {class_labels[predicted_class]} | Confidence: {confidence:.2f} | Prediction Time: {prediction_time:.3f}s")

    print("=====================================================================================")

    return class_labels[predicted_class]



def process_images_in_directory(test_images_dir):
    results = []
    
    for filename in os.listdir(test_images_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  
            image_path = os.path.join(test_images_dir, filename)
            
            predicted_class_name = process_single_image(image_path)
            
            results.append(f"{filename}: {predicted_class_name}")
    
    for result in results:
        print(result)


if __name__ == "__main__":
    run()