import tensorflow as tf
import numpy as np
import cv2
import os
import sys

def detect_pneumonia(model_path, img_path):
    #load the model
    pneumonia_model = tf.keras.models.load_model(model_path)
    
    #read image and preprocess
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not read image from {img_path}")
        return
    img = cv2.resize(img, (150, 150))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    #predict
    prob = pneumonia_model.predict(img)[0][0]
    label = "PNEUMONIA" if prob > 0.5 else "NORMAL"
    print(label)


model_path = os.path.join(os.path.dirname(__file__), 'pneumonia_detection_model.h5')
detect_pneumonia(model_path,"./uploads/Untitled.jpeg")
