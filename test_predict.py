import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import time

# Load model
model = tf.keras.models.load_model('models/best_model.keras')
print("Model loaded.")

# Test image (use an actual test image path)
img_path = 'data/test/Glioma/1409.jpg'  # adjust if needed
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = preprocess_input(np.expand_dims(img_array, axis=0))

print("Predicting...")
start = time.time()
pred = model.predict(img_array, verbose=0)
print(f"Prediction took {time.time() - start:.2f} seconds.")
print("Prediction:", pred)