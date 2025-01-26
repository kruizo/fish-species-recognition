from flask import Blueprint, request, jsonify
from backend.models.baseline_resnet50 import BASELINE_RESNET50
from backend.models.esrgan import ESRGAN
from backend.utils.helpers import preprocess_image_for_classifier, save_image

backend = Blueprint("backend", __name__)
baseline_loader = BASELINE_RESNET50()
esrgan_loader = ESRGAN()

test_images_dir = 'test/images'
output_dir = 'test/results'

class_labels = [
    'Abudefduf Vaigiensis', 'Acanthurus Nigrofuscus', 'Balistapus Undulatus', 'Canthigaster Valentini', 
    'Chaetodon Trifascialis', 'Hemigymnus Fasciatus', 'Hemigymnus Melapterus', 'Lutjanus fulvus', 
    'Myripristis Kuntee', 'Neoglyphidodon Nigroris', 'Neoniphon Sammara', 'Pempheris Vanicolensis', 
    'Pomacentrus Moluccensis', 'Scaridae', 'Scolopsis Bilineata', 'Siganus Fuscescens', 'Zanclus Cornutus', 
    'Zebrasoma Scopas'
]

@backend.route("/predict", methods=["POST"])
def predict_endpoint():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    enhanced_image = esrgan_loader.enhance_image(file)
    
    # Save the enhanced image
    filename = file.filename
    print(f"Saving image: {filename}")
    save_image(enhanced_image, filename, output_dir)
    print(f"Image saved: {filename}")
    
    # Preprocess the image for classification
    processed_image = preprocess_image_for_classifier(enhanced_image)

    # Classify the image using baseline ResNet50
    predicted_class = baseline_loader.predict(processed_image)
    predicted_class_name = class_labels[predicted_class]
    print(f"Prediction: {predicted_class_name}")

    return jsonify({"prediction": predicted_class_name})

def setup_routes(app):
    app.register_blueprint(backend)
