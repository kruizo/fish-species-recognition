import os
import sys
import warnings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from flask import Blueprint, request, jsonify
from backend.models.baseline_resnet50 import BASELINE_RESNET50
from backend.models.unet import UNET
from backend.models.esrgan import ESRGAN

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from backend.utils.helpers import save_image

api = Blueprint("backend", __name__)

@api.route("/predict", methods=["POST"])
def predict_endpoint():
    class_labels = [
    'Abudefduf Vaigiensis', 'Acanthurus Nigrofuscus', 'Balistapus Undulatus', 'Canthigaster Valentini', 
    'Chaetodon Trifascialis', 'Hemigymnus Fasciatus', 'Hemigymnus Melapterus', 'Lutjanus fulvus', 
    'Myripristis Kuntee', 'Neoglyphidodon Nigroris', 'Neoniphon Sammara', 'Pempheris Vanicolensis', 
    'Pomacentrus Moluccensis', 'Scaridae', 'Scolopsis Bilineata', 'Siganus Fuscescens', 'Zanclus Cornutus', 
    'Zebrasoma Scopas'
    ]

    output_dir = 'test/results'

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if not file:
        return jsonify({"error": "No file to be processed"}), 400

    print('Loading image:')
    image = Image.open(file)
    print('Image loaded...')
    print('Enhancing image...')
    enhanced_image = ESRGAN().enhance_image(image)
    print('Image enhanced...')

    print('Masking image...')
    masked_image = UNET().mask(enhanced_image)
    print('Image masked...')
    
    filename = os.path.basename(file.filename)
    print(f"Saving Enhanced image: {filename}")
    save_image(enhanced_image, {"enhanced - " + filename}, output_dir)
    save_image(masked_image, {"masked - " + filename}, output_dir)
    print(f"Image saved: {filename}")
    
    print("Classifying the image...")
    predicted_class, confidence, prediction_time = BASELINE_RESNET50().predict(masked_image)
    predicted_class_name = class_labels[predicted_class]
    print(f"Prediction: {predicted_class_name} | Confidence: {confidence} | Prediction Time: {prediction_time}")
    
    return jsonify({"prediction": predicted_class_name, "confidence": confidence, "prediction_time": prediction_time, "image": filename}), 200

def setup_routes(app):
    app.register_blueprint(api)
