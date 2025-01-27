from flask import Blueprint, request, jsonify
from backend.models.baseline_resnet50 import BASELINE_RESNET50
from backend.models.esrgan import ESRGAN
from backend.utils.helpers import preprocess_image_for_classifier, save_image

api = Blueprint("backend", __name__)

@api.route("/predict", methods=["POST"])
def predict_endpoint():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    return jsonify({"prediction": ""})

def setup_routes(app):
    app.register_blueprint(api)
