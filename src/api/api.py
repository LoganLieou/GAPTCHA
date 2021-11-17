import os
from flask import Flask, request
from werkzeug.utils import secure_filename
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)
CORS(app)

# load the model in
model = tf.keras.models.load_model("./control_model")

valid_characters = "0123456789abcdefghijklmnopqrstuvwxyz"

def predict(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = img / 255.0
    else:
        print("Not detected");
    res = np.array(model.predict(img[np.newaxis, :, :, np.newaxis]))
    ans = np.reshape(res, (5, 36))
    l_ind = []
    probs = []
    for a in ans:
        l_ind.append(np.argmax(a))

    capt = ''
    for l in l_ind:
        capt += valid_characters[l]
    return capt

@app.route("/upload", methods=["POST", "GET"])
def upload_file():
    if (request.method == "POST"):
        if "file" not in request.files:
            print("ERROR")
            return "ERROR"
        else:
            file = request.files["file"]
            filename = secure_filename(file.filename)
            file.save(os.path.join("UPLOAD/", filename))

            # return the prediction
            return predict(os.path.join("UPLOAD/", filename))

