import base64
from io import BytesIO
import os
import sys
import time
import warnings
import numpy as np
import torch
from flask import Blueprint, request, jsonify
from PIL import Image, ImageFile
# Models
from backend.models.baseline import BASELINE_RESNET50
from backend.models.proposed import PROPOSED_RESNET50
from backend.models.inception import INCEPTIONV3
from backend.models.densenet import DENSENET121
from backend.models.mobilenet import MOBILENET
from backend.models.unet import UNET
from backend.models.esrgan import ESRGAN

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ImageFile.LOAD_TRUNCATED_IMAGES = True

api = Blueprint("backend", __name__)

class_labels = [
'Abudefduf vaigiensis', 'Acanthurus nigrofuscus', 'Balistapus undulatus', 'Canthigaster valentini', 
'Chaetodon trifascialis', 'Hemigymnus fasciatus', 'Hemigymnus melapterus', 'Lutjanus fulvus', 
'Myripristis kuntee', 'Neoglyphidodon Nigroris', 'Neoniphon Sammara', 'Pempheris Vanicolensis', 
'Pomacentrus moluccensis', 'Scaridae', 'Scolopsis bilineata', 'Siganus fuscescens', 'Zanclus cornutus', 
'Zebrasoma scopas'
]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# vgg_model = VGG16(device=device)
inception_model = INCEPTIONV3(device=device)
densenet_model = DENSENET121(device=device)
mobilenet_model = MOBILENET(device=device)
unet_model = UNET(device=device)
esrgan_model = ESRGAN(device=device)
proposed_model = PROPOSED_RESNET50(device=device)
baseline_model = BASELINE_RESNET50(device=device)

@api.route('/predict/model/mobilenet', methods=['POST'])
def predict_mobilenet_endpoint():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if not file:
        return jsonify({"error": "No file to be processed"}), 400

    file = request.files["file"]
    image = Image.open(file)

    enhanced_image, esrgan_time = esrgan_model.predict(image)
    
    binary_mask, unet_time = unet_model.predict(enhanced_image)

    enhanced_image = enhanced_image.squeeze().cpu().numpy().transpose(1, 2, 0)

    masked_image = enhanced_image * np.expand_dims(binary_mask, axis=-1)
    masked_image = np.clip(masked_image * 255, 0, 255).astype(np.uint8)
    

    predicted_class, confidence, prediction_time, probabilities = mobilenet_model.predict(masked_image)

    total_time = prediction_time + esrgan_time + unet_time

    buffered = BytesIO()
    Image.fromarray(masked_image).save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    img_str = f"data:image/png;base64,{img_str}"

    print(f"MOBILENET= Prediction: {class_labels[predicted_class]} | Confidence: {confidence:.2f} | Prediction Time: {prediction_time:.3f}s | Total Time: {total_time:.3f}s")
    return jsonify({
        "name": 'densenet121',
        "image": img_str,
        "prediction": class_labels[predicted_class],
        "confidence": confidence,
        "prediction_time": prediction_time,
        "probabilities": probabilities.tolist()
    }), 200


@api.route('/predict/model/inception', methods=['POST'])
def predict_inception_endpoint():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if not file:
        return jsonify({"error": "No file to be processed"}), 400

    file = request.files["file"]
    image = Image.open(file)

    enhanced_image, esrgan_time = esrgan_model.predict(image)
    
    binary_mask, unet_time = unet_model.predict(enhanced_image)

    enhanced_image = enhanced_image.squeeze().cpu().numpy().transpose(1, 2, 0)

    masked_image = enhanced_image * np.expand_dims(binary_mask, axis=-1)
    masked_image = np.clip(masked_image * 255, 0, 255).astype(np.uint8)

    predicted_class, confidence, prediction_time, probabilities = inception_model.predict(masked_image)

    total_time = prediction_time + esrgan_time + unet_time

    buffered = BytesIO()
    Image.fromarray(masked_image).save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    img_str = f"data:image/png;base64,{img_str}"

    print(f"INCEPTION= Prediction: {class_labels[predicted_class]} | Confidence: {confidence:.2f} | Prediction Time: {prediction_time:.3f}s | Total Time: {total_time:.3f}s")
    return jsonify({
        "name": 'densenet121',
        "image": img_str,
        "prediction": class_labels[predicted_class],
        "confidence": confidence,
        "prediction_time": prediction_time,
        "probabilities": probabilities.tolist()
    }), 200

@api.route('/predict/model/densenet', methods=['POST'])
def predict_densenet_endpoint():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if not file:
        return jsonify({"error": "No file to be processed"}), 400

    file = request.files["file"]
    image = Image.open(file)
    

    enhanced_image, esrgan_time = esrgan_model.predict(image)
    
    binary_mask, unet_time = unet_model.predict(enhanced_image)

    enhanced_image = enhanced_image.squeeze().cpu().numpy().transpose(1, 2, 0)

    masked_image = enhanced_image * np.expand_dims(binary_mask, axis=-1)
    masked_image = np.clip(masked_image * 255, 0, 255).astype(np.uint8)
    

    predicted_class, confidence, prediction_time, probabilities = densenet_model.predict(masked_image)

    total_time = prediction_time + esrgan_time + unet_time

    buffered = BytesIO()
    Image.fromarray(masked_image).save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    img_str = f"data:image/png;base64,{img_str}"

    print(f"DENSENET= Prediction: {class_labels[predicted_class]} | Confidence: {confidence:.2f} | Prediction Time: {prediction_time:.3f}s | Total Time: {total_time:.3f}s")
    return jsonify({
        "name": 'densenet121',
        "image": img_str,
        "prediction": class_labels[predicted_class],
        "confidence": confidence,
        "prediction_time": prediction_time,
        "probabilities": probabilities.tolist()
    }), 200

@api.route("/predict", methods=["POST"])
def predict_endpoint():

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if not file:
        return jsonify({"error": "No file to be processed"}), 400

    print('Image:', file)
    image = Image.open(file)

    bpredicted_class, bconfidence, bprediction_time, bprobabilities = baseline_model.predict(image)

    enhanced_image, esrgan_time = esrgan_model.predict(image)
    
    binary_mask, unet_time = unet_model.predict(enhanced_image)

    enhanced_image = enhanced_image.squeeze().cpu().numpy().transpose(1, 2, 0)

    masked_image = enhanced_image * np.expand_dims(binary_mask, axis=-1)
    masked_image = np.clip(masked_image * 255, 0, 255).astype(np.uint8)
    
    predicted_class, confidence, prediction_time, probabilities = proposed_model.predict(masked_image)


    total_time = prediction_time + esrgan_time + unet_time

    print(f"BASELINE Prediction: {class_labels[bpredicted_class]} | Confidence: {bconfidence:.2f} | Prediction Time: {bprediction_time:.3f}s")
    print(f"PROPOSED Prediction: {class_labels[predicted_class]} | Confidence: {confidence:.2f} | Prediction Time: {prediction_time:.3f}s | ESRGAN Time: {esrgan_time:.3f} | UNET Time: {unet_time:.3f} | Total Time: {total_time:.3f}" )
    print("=====================================================================================")

    buffered = BytesIO()
    Image.fromarray(masked_image).save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    img_str = f"data:image/png;base64,{img_str}"

    original_buffered = BytesIO()
    image.save(original_buffered, format="PNG")
    original_img_str = base64.b64encode(original_buffered.getvalue()).decode('utf-8')
    original_img_str = f"data:image/png;base64,{original_img_str}"

    return jsonify({
        "class_labels": class_labels,
        "models" : {
            'baseline' : {
                "prediction": class_labels[bpredicted_class],
                "confidence": bconfidence,
                "prediction_time": bprediction_time,
                "probabilities" : bprobabilities.tolist(),
                "image" : original_img_str
                },
            'proposed' : {
                "prediction": class_labels[predicted_class],
                "confidence": confidence,
                "prediction_time": total_time,
                "probabilities": probabilities.tolist(),
                "image" : img_str
            }
        }
    }), 200

def setup_routes(app):
    app.register_blueprint(api)
