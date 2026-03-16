"""
debug_gradcam.py — Run this from your project root to find the exact Grad-CAM error.

    python debug_gradcam.py
"""

import sys
import traceback
import numpy as np
import tensorflow as tf

print(f"Python : {sys.version}")
print(f"TF     : {tf.__version__}")
print()

# ── 1. Load model ─────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading model")
print("=" * 60)
try:
    model = tf.keras.models.load_model(
        "models/brain_tumor_vgg16.keras", compile=False
    )
    print("✅ Model loaded OK")
except Exception:
    print("❌ Model load FAILED:")
    traceback.print_exc()
    sys.exit(1)

# ── 2. Inspect model.inputs ───────────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 2: Inspecting model inputs/layers")
print("=" * 60)
print(f"  type(model.inputs)    : {type(model.inputs)}")
print(f"  len(model.inputs)     : {len(model.inputs)}")
print(f"  model.inputs[0]       : {model.inputs[0]}")
print(f"  type(model.inputs[0]) : {type(model.inputs[0])}")
print()
print(f"  model.layers[0]       : {model.layers[0]}")
print(f"  model.layers[0].name  : {model.layers[0].name}")
try:
    print(f"  model.input           : {model.input}")
    print(f"  type(model.input)     : {type(model.input)}")
except Exception as e:
    print(f"  model.input FAILED    : {e}")

# ── 3. Find last conv layer ───────────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 3: Finding last Conv2D layer")
print("=" * 60)
from src.gradcam import find_last_conv_layer
target = find_last_conv_layer(model)
print(f"  target layer name : {target.name}")
print(f"  target layer type : {type(target)}")
try:
    print(f"  target.output     : {target.output}")
except Exception as e:
    print(f"  target.output FAILED: {e}")

# ── 4. Try building grad_model ────────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 4: Building grad sub-model")
print("=" * 60)

# Attempt A: model.inputs[0]
print("  Attempt A — tf.keras.Model(inputs=model.inputs[0], ...)")
try:
    gm = tf.keras.Model(
        inputs=model.inputs[0],
        outputs=[target.output, model.output],
    )
    print("  ✅ Attempt A succeeded")
except Exception as e:
    print(f"  ❌ Attempt A FAILED: {e}")

# Attempt B: model.layers[0].output
print("  Attempt B — tf.keras.Model(inputs=model.layers[0].output, ...)")
try:
    gm = tf.keras.Model(
        inputs=model.layers[0].output,
        outputs=[target.output, model.output],
    )
    print("  ✅ Attempt B succeeded")
except Exception as e:
    print(f"  ❌ Attempt B FAILED: {e}")

# Attempt C: model.input
print("  Attempt C — tf.keras.Model(inputs=model.input, ...)")
try:
    gm = tf.keras.Model(
        inputs=model.input,
        outputs=[target.output, model.output],
    )
    print("  ✅ Attempt C succeeded")
except Exception as e:
    print(f"  ❌ Attempt C FAILED: {e}")

# ── 5. Run full Grad-CAM with traceback ───────────────────────────────────────
print()
print("=" * 60)
print("STEP 5: Running make_gradcam_heatmap end-to-end")
print("=" * 60)
from src.gradcam import make_gradcam_heatmap

dummy_img = np.random.rand(224, 224, 3).astype(np.float32)
try:
    heatmap = make_gradcam_heatmap(dummy_img, model, pred_index=0)
    print(f"✅ Grad-CAM succeeded — heatmap shape: {heatmap.shape}")
except Exception:
    print("❌ Grad-CAM FAILED — full traceback:")
    traceback.print_exc()