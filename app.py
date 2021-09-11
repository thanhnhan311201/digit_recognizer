from flask import Flask, render_template, request, jsonify

import cv2
import numpy as np 
import tensorflow as tf
from tensorflow import keras
from src.digit_classifier_cnn import *

app = Flask(__name__, template_folder='templates', static_folder='static')

# model = keras.models.load_model('models/cnn_model.h5')
# digit_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

@app.route('/ann')
def digit_recog_ann():
    return render_template('ann.html')

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/cnn')
def digit_recog_cnn():
    return render_template('cnn.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        image_file = file.read()
        image = cv2.imdecode(np.frombuffer(image_file, dtype=np.uint8), -1)
        res = str(predict_image(image))
        
        return res

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)