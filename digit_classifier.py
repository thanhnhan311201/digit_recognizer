from tensorflow import keras
import tensorflow as tf

import cv2
import numpy as np
import glob2
import os

def read_image_path(path):
    all_images = []
    for ext in ["*.jpg",  "*.jpeg", "*.png"]:
        image = glob2.glob(os.path.join(path, ext))
        all_images += image
    
    return all_images

def image_processing(img):
    resized_img = cv2.resize(img, (28, 28), interpolation = cv2.INTER_AREA)
    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    (thresh, bw_img) = cv2.threshold(gray_img, 127, 1, cv2.THRESH_BINARY_INV)
    bw_img = np.array([np.array(bw_img.flatten())])
    
    return bw_img

def create_model():
    model = keras.models.Sequential()

    model.add(keras.layers.Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',  activation ='relu', input_shape = (28,28,1)))
    model.add(keras.layers.Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
    model.add(keras.layers.Dropout(0.25))


    model.add(keras.layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
    model.add(keras.layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(keras.layers.Dropout(0.25))


    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation = "relu"))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(10, activation = "softmax"))

    return model

def predict_image(img):
    model = create_model()
    model = keras.models.load_model('models/cnn_model.h5')

    img = image_processing(img)
    res = model.predict(img)
    res = np.argmax(res, axis=1)
    
    return res[0]

if __name__ == '__main__':
    in_path = 'input/'
    all_images = read_image_path(in_path)
    
    img = np.array(cv2.imread('input/img_9.png'))
    res = predict_image(img)

    print(res)
    
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()