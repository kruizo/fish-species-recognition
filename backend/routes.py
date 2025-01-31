import base64
from io import BytesIO
import os
import sys
import warnings
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from flask import Blueprint, request, jsonify
from backend.models.baseline import BASELINE_RESNET50
from backend.models.proposed import PROPOSED_RESNET50
from backend.models.unet import UNET
from backend.models.esrgan import ESRGAN
from backend.utils.helpers import convert_img_numpy, conver_mask_numpy, save_image, save_image_as_png

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


api = Blueprint("backend", __name__)

class_labels = [
'Abudefduf Vaigiensis', 'Acanthurus Nigrofuscus', 'Balistapus Undulatus', 'Canthigaster Valentini', 
'Chaetodon Trifascialis', 'Hemigymnus Fasciatus', 'Hemigymnus Melapterus', 'Lutjanus fulvus', 
'Myripristis Kuntee', 'Neoglyphidodon Nigroris', 'Neoniphon Sammara', 'Pempheris Vanicolensis', 
'Pomacentrus Moluccensis', 'Scaridae', 'Scolopsis Bilineata', 'Siganus Fuscescens', 'Zanclus Cornutus', 
'Zebrasoma Scopas'
]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@api.route("/predict", methods=["POST"])
def predict_endpoint():



    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if not file:
        return jsonify({"error": "No file to be processed"}), 400

    print('Image:', file)
    image = Image.open(file)
    # print('Image loaded...')

    bpredicted_class, bconfidence, bprediction_time = BASELINE_RESNET50(device=device).predict(image)


    # print('Enhancing image...')
    enhanced_image, time_taken = ESRGAN(device=device).predict(image)
    # print('Image enhanced... TIME TAKEN: ', time_taken)
    # print('Image enhanced SHAPE:', enhanced_image.shape)
    # print(f"Saving Enhanced image: {filename}")
    
    # save_image_as_png(enhanced_image, "enhanced - " + filename, enhanced_output_dir)
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
    # save_image(masked_image, "masked - " + filename, masked_output_dir)
    # print(f"Image saved: {filename}")
    
    # print("Classifying the image...")
    predicted_class, confidence, prediction_time = PROPOSED_RESNET50(device=device).predict(masked_image)
    print(f"BASELINE Prediction: {class_labels[bpredicted_class]} | Confidence: {bconfidence:.2f} | Prediction Time: {bprediction_time:.3f}s")
    print(f"PROPOSED Prediction: {class_labels[predicted_class]} | Confidence: {confidence:.2f} | Prediction Time: {prediction_time:.3f}s")

    print("=====================================================================================")

    buffered = BytesIO()
    Image.fromarray(masked_image).save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    
    return jsonify({
        "baseline_prediction": class_labels[bpredicted_class],
        "baseline_confidence": bconfidence,
        "baseline_prediction_time": bprediction_time,
        "proposed_prediction": class_labels[predicted_class],
        "proposed_confidence": confidence,
        "proposed_prediction_time": prediction_time,
        "masked_image": img_str
    }), 200

def setup_routes(app):
    app.register_blueprint(api)
