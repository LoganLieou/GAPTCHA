import os
from flask import Flask, request
from werkzeug.utils import secure_filename
from flask_cors import CORS
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# load the model in
model = tf.keras.models.load_model("ocr_model")

@app.route("upload/", methods=["POST", "GET"])
def upload_file():
    if (request.method == "POST"):
        if "file" not in request.files:
            print("ERROR")
            return "ERROR"
        else:
            file = request.files["file"]
            filename = secure_filename(file.filename)
            file.save(os.path.join("UPLOAD/", filename))

            """ AI STUFF
            prediction = model.predict(filename)
            return prediction
            """
