from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import cv2


# Keras
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Define a flask app
app = Flask(__name__)

# Setup model path
MODEL_PATH = 'resnet_model.h5'

# Load trained model
model = load_model(MODEL_PATH)

print('Model loaded. Check http://127.0.0.1:5000/')


def predict_disease(image_path, model):
    img = cv2.imread(image_path)
    resized = cv2.resize(img,(100,100))
    img_array = tf.keras.utils.img_to_array(resized)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    label_names = ['โรคใบจุดมะเขือเทศ (Bacterial Spot Disease)',
                    'โรคใบไหม้ (Early Blight Disease)',
                    'ไม่พบโรค (Healthy)',
                    'โรคใบจุดวงกลม (Septoria Leaf Spot Disease)',
                    'โรคใบจุด (Target Spot Disease)',
                    'โรคใบหงิกเหลือง (Tomato Yellow Leaf Curl Virus Disease)']
    label=label_names[np.argmax(score)]
    return label

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

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
        preds_result = predict_disease(file_path, model)

        return preds_result
    return None


if __name__ == '__main__':
    app.run(debug=True)