import os
import sys
import warnings
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
torch.cuda.empty_cache()

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from backend.models.esrgan import ESRGAN
from backend.models.unet import UNET
from backend.utils.helpers import save_image

class_labels = [
    'Abudefduf Vaigiensis', 'Acanthurus Nigrofuscus', 'Balistapus Undulatus', 'Canthigaster Valentini', 
    'Chaetodon Trifascialis', 'Hemigymnus Fasciatus', 'Hemigymnus Melapterus', 'Lutjanus fulvus', 
    'Myripristis Kuntee', 'Neoglyphidodon Nigroris', 'Neoniphon Sammara', 'Pempheris Vanicolensis', 
    'Pomacentrus Moluccensis', 'Scaridae', 'Scolopsis Bilineata', 'Siganus Fuscescens', 'Zanclus Cornutus', 
    'Zebrasoma Scopas'
]

output_dir = 'test/results/mask'

def run():
    image_path = 'test/images/fish_000065789596_04756.png'
    process_single_image(image_path, output_dir)


def process_single_image(image_path, output_dir):
    # Load image
    print('Loading image:', image_path)
    image = Image.open(image_path)

    
    image = np.clip(image, 0, 255).astype(np.uint8)
    
    image = Image.fromarray(image)
    
    print('Image loaded...')

    print('Masking original image...')
    masked_image = UNET().mask(image)
    print('Image masked...')

    print('Enhancing image...')
    enhanced_image = ESRGAN().enhance_image(image)
    print('Image enhanced...')

    print('Masking enhanced image...')
    masked_enhanced_image = UNET().mask(enhanced_image)
    print('Image masked...')

    filename = os.path.basename(image_path)
    print(f"Saving Enhanced image: {filename}")
    save_image(masked_enhanced_image, {"enhanced - " + filename}, output_dir)
    save_image(masked_image, {"original - " + filename}, output_dir)
    print(f"Image saved: {filename}")
    

if __name__ == "__main__":
    run()