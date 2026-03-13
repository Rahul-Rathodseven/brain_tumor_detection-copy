"""
gradcam.py — Grad-CAM visualisation for the VGG16 brain-tumor classifier.
"""

import cv2
import numpy as np
import tensorflow as tf

_CONV_TYPES = (
    tf.keras.layers.Conv2D,
    tf.keras.layers.SeparableConv2D,
    tf.keras.layers.DepthwiseConv2D,
)


def find_last_conv_layer(model):
    """Recursively return the last Conv2D-like layer in the model tree."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            found = find_last_conv_layer(layer)
            if found is not None:
                return found
        if isinstance(layer, _CONV_TYPES):
            return layer
    return None


def make_gradcam_heatmap(img_array, model, pred_index=None):
    """
    Compute Grad-CAM heatmap for a single image.

    Parameters
    ----------
    img_array  : np.ndarray shape (H,W,C) or (1,H,W,C)
    model      : Keras Functional model (loaded from .keras file)
    pred_index : int or None

    Returns
    -------
    np.ndarray shape (h,w), float32, values in [0,1]
    """
    if img_array.ndim == 3:
        img_array = np.expand_dims(img_array, axis=0)
    img_tensor = tf.cast(img_array, tf.float32)

    target_layer = find_last_conv_layer(model)
    if target_layer is None:
        raise ValueError("No Conv2D layer found in model.")

    # CRITICAL: use model.input (single tensor), NOT model.inputs (list)
    # model.inputs returns a list which causes:
    #   "list indices must be integers or slices, not tuple"
    # when the model is loaded from a .keras file.
    try:
        model_input = model.input       # primary — works for saved .keras models
    except AttributeError:
        model_input = model.inputs[0]   # fallback

    grad_model = tf.keras.Model(
        inputs=model_input,
        outputs=[target_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor, training=False)
        tape.watch(conv_outputs)
        if pred_index is None:
            pred_index = int(tf.argmax(predictions[0]))
        class_score = predictions[:, pred_index]

    gradients = tape.gradient(class_score, conv_outputs)
    if gradients is None:
        raise RuntimeError(f"Gradients are None for '{target_layer.name}'.")

    weights    = tf.reduce_mean(gradients, axis=(0, 1, 2))
    activation = conv_outputs[0]
    heatmap    = tf.reduce_sum(activation * weights, axis=-1)
    heatmap    = tf.nn.relu(heatmap)

    max_val = float(tf.reduce_max(heatmap))
    if max_val == 0.0:
        return np.zeros(heatmap.shape, dtype=np.float32)

    return (heatmap / max_val).numpy().astype(np.float32)


def overlay_heatmap(heatmap, original_img, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """Blend Grad-CAM heatmap onto the original image."""
    alpha = float(np.clip(alpha, 0.0, 1.0))

    if hasattr(original_img, "convert"):
        original_img = np.array(original_img.convert("RGB"))
    original_img = np.asarray(original_img)
    if original_img.ndim == 2:
        original_img = np.stack([original_img] * 3, axis=-1)
    if original_img.dtype != np.uint8:
        scale = 255.0 if original_img.max() <= 1.0 else 1.0
        original_img = np.clip(original_img * scale, 0, 255).astype(np.uint8)

    heatmap = np.nan_to_num(np.clip(np.squeeze(heatmap), 0.0, 1.0))
    h, w    = original_img.shape[:2]
    heatmap = cv2.resize(heatmap, (w, h))
    lo, hi  = float(heatmap.min()), float(heatmap.max())
    if hi > lo:
        heatmap = (heatmap - lo) / (hi - lo)
    heatmap = np.power(heatmap, 0.7)

    heatmap_u8     = np.uint8(255 * heatmap)
    heatmap_colour = cv2.applyColorMap(heatmap_u8, colormap)
    heatmap_colour = cv2.cvtColor(heatmap_colour, cv2.COLOR_BGR2RGB)

    alpha_mask = heatmap[..., None] * alpha
    blended    = (
        original_img.astype(np.float32) * (1.0 - alpha_mask)
        + heatmap_colour.astype(np.float32) * alpha_mask
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


__all__ = ["find_last_conv_layer", "make_gradcam_heatmap", "overlay_heatmap"]