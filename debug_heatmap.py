"""
debug_heatmap.py — Diagnose why the Grad-CAM heatmap is invisible.

Usage:
    python debug_heatmap.py path/to/any_mri.jpg
"""

import sys
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

from src.gradcam import find_last_conv_layer, make_gradcam_heatmap, overlay_heatmap

IMG_PATH = sys.argv[1] if len(sys.argv) > 1 else None
if not IMG_PATH:
    print("Usage: python debug_heatmap.py path/to/image.jpg")
    sys.exit(1)

print("Loading model...")
model = tf.keras.models.load_model("models/brain_tumor_vgg16.keras", compile=False)
_ = model(tf.zeros((1, 224, 224, 3)), training=False)
print("Model loaded.")

# Load and preprocess image
img = keras_image.load_img(IMG_PATH, target_size=(224, 224))
arr = preprocess_input(keras_image.img_to_array(img))
arr = np.expand_dims(arr, 0).astype(np.float32)

# Run prediction
probs = model.predict(arr, verbose=0)[0]
pred_index = int(probs.tolist().index(max(probs.tolist())))
print(f"\nPrediction: class {pred_index}, probs={probs}")

# Get raw heatmap
img_squeezed = np.squeeze(arr, axis=0)
heatmap = make_gradcam_heatmap(img_squeezed, model, pred_index)

print(f"\n--- RAW HEATMAP STATS ---")
print(f"  shape      : {heatmap.shape}")
print(f"  dtype      : {heatmap.dtype}")
print(f"  min        : {heatmap.min():.6f}")
print(f"  max        : {heatmap.max():.6f}")
print(f"  mean       : {heatmap.mean():.6f}")
print(f"  std        : {heatmap.std():.6f}")
print(f"  % positive : {(heatmap > 0).mean()*100:.1f}%")
print(f"  % zero     : {(heatmap == 0).mean()*100:.1f}%")
print(f"  % negative : {(heatmap < 0).mean()*100:.1f}%")
print(f"  p50        : {np.percentile(heatmap, 50):.6f}")
print(f"  p90        : {np.percentile(heatmap, 90):.6f}")
print(f"  p99        : {np.percentile(heatmap, 99):.6f}")

# Save raw heatmap as standalone image (no blending) to see if it has signal
h_norm = heatmap.copy()
h_norm = h_norm - h_norm.min()
if h_norm.max() > 1e-8:
    h_norm = h_norm / h_norm.max()
h_u8 = np.uint8(255 * h_norm)
h_resized = cv2.resize(h_u8, (224, 224))
h_colour = cv2.applyColorMap(h_resized, cv2.COLORMAP_JET)
cv2.imwrite("debug_heatmap_raw.png", h_colour)
print(f"\nSaved raw heatmap (no blend) → debug_heatmap_raw.png")
print("Open this file — if it shows colour variation, the heatmap IS working")
print("and the issue is only in the blending with the MRI image.")

# Also save the overlay at alpha=0.7
from PIL import Image
pil_img = Image.open(IMG_PATH).convert("RGB")
overlay = overlay_heatmap(heatmap, np.array(pil_img), alpha=0.7)
overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
cv2.imwrite("debug_overlay_alpha07.png", overlay_bgr)
print(f"Saved overlay at alpha=0.7 → debug_overlay_alpha07.png")
