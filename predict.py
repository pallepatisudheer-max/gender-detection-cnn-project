import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import sys

model = tf.keras.models.load_model('models/gender_model.h5')

def predict_gender(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)

    if prediction[0][0] > 0.5:
        gender = "Male"
        confidence = prediction[0][0] * 100
    else:
        gender = "Female"
        confidence = (1 - prediction[0][0]) * 100

    return gender, confidence

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: py predict.py <image_path>")
    else:
        gender, confidence = predict_gender(sys.argv[1])
        print(f"Prediction: {gender}")
        print(f"Confidence: {confidence:.2f}%")