import os
import io
import re
import base64

from PIL import Image
from flask import Flask, request

from tf_model_helper import TFModel

app = Flask(__name__)

# Path to signature.json and model file
ASSETS_PATH = os.path.join(".", "./model")
TF_MODEL = TFModel(ASSETS_PATH)


@app.post('/predict')
def predict_image():
    req = request.get_json(force=True)
    image = _process_base64(req)
    return TF_MODEL.predict(image)

def _process_base64(json_data):
    image_data = json_data.get("image")
    image_data = re.sub(r"^data:image/.+;base64,", "", image_data)
    image_base64 = bytearray(image_data, "utf8")
    image = base64.decodebytes(image_base64)
    return Image.open(io.BytesIO(image))


@app.route("/")
def index():
    return "Hello this is the new version!"


@app.route("/predict")
def predict():
    return "predict"
