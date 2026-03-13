"""
Run this script once to find the correct CLASSES order for your saved model.

Usage:
    python check_model_classes.py

It will print the exact CLASSES list you need to put in config.py.
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'brain_tumor_vgg16.keras')
model = load_model(MODEL_PATH, compile=False)
print(f"Model loaded from {MODEL_PATH}")
print(f"Output shape: {model.output_shape}")
print(f"Number of classes: {model.output_shape[-1]}\n")

# ── Test with known images ────────────────────────────────────────────────────
# Point these to any image you KNOW the class of from your data/test folder
TEST_IMAGES = {
    # format: "true_class_name" : "path/to/image.jpg"
    # Fill in real paths from your data/test directory
    "notumor"   : "data/test/No_Tumor",    # adjust folder name if needed
    "pituitary" : "data/test/Pituitary",
    "glioma"    : "data/test/Glioma",
    "meningioma": "data/test/Meningioma",
}

print("Testing one image per class to find correct label mapping:")
print("─" * 55)

for true_class, folder in TEST_IMAGES.items():
    if not os.path.isdir(folder):
        print(f"  Folder not found: {folder}")
        continue

    # Get first image in folder
    files = [f for f in sorted(os.listdir(folder))
             if f.lower().endswith(('.jpg','.jpeg','.png'))]
    if not files:
        continue

    img_path = os.path.join(folder, files[0])
    img = image.load_img(img_path, target_size=(224, 224))
    arr = preprocess_input(image.img_to_array(img))
    probs = model.predict(arr[np.newaxis], verbose=0)[0]

    print(f"\n  True class : {true_class}")
    print(f"  Image      : {files[0]}")
    print(f"  Raw probs  : {[f'{p:.4f}' for p in probs]}")
    print(f"  Argmax idx : {np.argmax(probs)}")
    print(f"  → Index {np.argmax(probs)} should map to '{true_class}'")

print("\n─" * 55)
print("\nBased on above, your CLASSES list should be:")
print("(index 0 = first prob, index 1 = second prob, etc.)\n")
print("CLASSES = [???]  ← fill in based on the idx→class mapping above")
