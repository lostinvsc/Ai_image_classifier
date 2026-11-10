import os
from io import BytesIO

import cv2
import numpy as np
import requests
from PIL import Image as PILImage
from tensorflow.keras.models import load_model

IMG_SIZE = 299

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inceptionv3f.weights.h5")
model = load_model(MODEL_PATH, compile=False)


def preprocess(pil_img):
    
    rgb = np.array(pil_img.convert("RGB"))               # (H, W, 3) in RGB, drops alpha/grayscale issues
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)           # match training channel order
    bgr = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE))          # same resize as training (INTER_LINEAR)
    arr = bgr.astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)                   # (1, 299, 299, 3)


def _predict(pil_img):
    prediction = model.predict(preprocess(pil_img), verbose=0)
    label = 1 if prediction[0] > 0.5 else 0              # 1 = real, 0 = AI-generated
    return label, prediction[0]


def predict_imagefromfile(image):
    """image: a PIL image (from the Streamlit uploader)."""
    return _predict(image)


def predict_image(image_url):
    """image_url: a link to an image."""
    response = requests.get(image_url, timeout=15)
    response.raise_for_status()
    img = PILImage.open(BytesIO(response.content))
    return _predict(img)


def final_resultURL(image_url):
    result, p = predict_image(image_url)
    print("real" if result == 1 else "fake")
    print("probability of real_image is ", p.item())


def final_resultPath(image_path="img1- (30).jpg"):
    img = PILImage.open(image_path)
    result, p = predict_imagefromfile(img)
    print("real" if result == 1 else "fake")
    print("probability of real_image is ", p.item())