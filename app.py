from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import os

app = Flask(__name__)
model = load_model("model.h5")

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
    if request.method == "POST":
        file = request.files["image"]
        if file:
            file_path = os.path.join("static", file.filename)
            file.save(file_path)
            prediction, confidence = predict_mineral(file_path)
            return render_template("index.html", prediction=prediction, confidence=confidence, image_path=file_path)
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
