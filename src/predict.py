"""
predict.py — Load, preprocess and classify a single MRI image.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

from src.config import BEST_MODEL_PATH, CLASSES, IMG_SIZE


def load_and_preprocess(img_path: str) -> np.ndarray:
    """
    Load an image from disk and apply VGG16 preprocessing.

    Returns
    -------
    np.ndarray of shape (1, 224, 224, 3), dtype float32
    """
    img = keras_image.load_img(img_path, target_size=IMG_SIZE)
    arr = keras_image.img_to_array(img)           # (224, 224, 3)
    arr = preprocess_input(arr)                    # VGG16 channel-mean subtraction
    return np.expand_dims(arr, axis=0).astype(np.float32)  # (1, 224, 224, 3)


def predict_image(model, img_path: str):
    """
    Run inference on a single image.

    Returns
    -------
    predicted_class : str   — internal class name e.g. 'notumor'
    confidence      : float — probability of predicted class
    probs           : np.ndarray shape (NUM_CLASSES,) — all class probabilities
                      ordered to match CLASSES list from config.py
    """
    preprocessed = load_and_preprocess(img_path)
    probs        = model.predict(preprocessed, verbose=0)[0]

    pred_idx         = int(np.argmax(probs))
    predicted_class  = CLASSES[pred_idx]
    confidence       = float(probs[pred_idx])

    return predicted_class, confidence, probs


def load_model_for_inference(model_path: str = BEST_MODEL_PATH):
    """Load saved model and run a warm-up pass."""
    model = tf.keras.models.load_model(model_path, compile=False)
    _     = model(tf.zeros((1, *IMG_SIZE, 3)), training=False)
    return model