from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import os

app = Flask(__name__)

# Assurez-vous que le modèle est bien dans le dossier actuel
model = load_model("model.h5")

# Classes de minéraux
class_labels = ['Alunite', 'Amethyst', 'Chalcedony', 'Cinnabar', 'Galena', 'Limonite', 'Malachite', 'Pyrite', 'Quartz']
IMG_SIZE = (224, 224)

def predict_mineral(img_path):
    img = Image.open(img_path).convert('RGB').resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)
    return class_labels[class_index], confidence * 100

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_path = None
    if request.method == "POST":
        file = request.files["image"]
        if file:
            # Assurez-vous que le dossier static existe
            os.makedirs("static", exist_ok=True)
            file_path = os.path.join("static", file.filename)
            file.save(file_path)
            prediction, confidence = predict_mineral(file_path)
            image_path = file_path
    return f"""
    <html>
    <head><title>Mineral Classifier</title></head>
    <body>
        <h1>Upload a Mineral Image</h1>
        <form method="POST" enctype="multipart/form-data">
            <input type="file" name="image">
            <input type="submit" value="Predict">
        </form>
        {'<h2>Prediction: {}</h2><h3>Confidence: {:.2f}%</h3><img src="{}" width="300">'.format(prediction, confidence, image_path) if prediction else ''}
    </body>
    </html>
    """

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

