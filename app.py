from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.utils import load_img, img_to_array



# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'combined_model2.h5'

# Load your trained model
model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')
print('Model loaded. Check http://127.0.0.1:5000/ or http://localhost:5000/')


def model_predict(img_path, model):

    img_pred = load_img(img_path, target_size = (100, 100))
    img_pred = img_to_array(img_pred)
    img_pred = np.expand_dims(img_pred, axis = 0)
    preds = model.predict(img_pred)

    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/covid')
def cov():
    return render_template('next.html')

@app.route('/pneumonia')
def neu():
    return render_template('next2.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        if preds[0][0] == 1:
            result = "COVID"
        elif preds[0][1] == 1:
            result = "NORMAL"
        else:
            result = "PNEUMONIA"

        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)