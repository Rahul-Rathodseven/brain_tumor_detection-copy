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
    """Return the last Conv2D-like layer in the model."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            found = find_last_conv_layer(layer)
            if found is not None:
                return found
        if isinstance(layer, _CONV_TYPES):
            return layer
    return None


def find_logit_layer(model):
    """
    Return the very last Dense layer in the model.

    In this model, the final Dense layer (`dense_1`) includes the softmax
    activation, so `layer.output` is already post-softmax. Grad-CAM needs the
    pre-activation class scores, which we reconstruct from this layer's input
    and weights inside `make_gradcam_heatmap`.
    """
    last_dense = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            last_dense = layer
    return last_dense


def _compute_pre_activation_scores(features, dense_layer):
    """Return the Dense layer's affine output before any activation is applied."""
    kernel = tf.cast(dense_layer.kernel, tf.float32)
    scores = tf.linalg.matmul(features, kernel)
    if dense_layer.use_bias:
        scores = tf.nn.bias_add(scores, tf.cast(dense_layer.bias, tf.float32))
    return scores


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
    np.ndarray shape (h,w), float32 raw values (not normalised)
    """
    if img_array.ndim == 3:
        img_array = np.expand_dims(img_array, axis=0)

    if pred_index is not None:
        pred_index = int(pred_index)

    target_layer = find_last_conv_layer(model)   # block5_conv3
    if target_layer is None:
        raise ValueError("No Conv2D layer found in model.")

    logit_layer = find_logit_layer(model)         # dense_1 (last Dense)
    if logit_layer is None:
        score_tensor = model.output
        layer_name = "model.output"
    else:
        score_tensor = logit_layer.input
        layer_name = f"{logit_layer.name}.pre_activation"

    print(f"[gradcam] conv={target_layer.name}  logit={layer_name}")

    grad_model = tf.keras.Model(
        inputs=model.inputs[0],
        outputs=[target_layer.output, score_tensor],
    )

    img_tensor = tf.constant(img_array, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        outputs      = grad_model(img_tensor, training=False)
        conv_outputs = tf.cast(tf.convert_to_tensor(outputs[0]), tf.float32)
        score_input  = tf.cast(tf.convert_to_tensor(outputs[1]), tf.float32)

        if logit_layer is None:
            logits = score_input
        else:
            logits = _compute_pre_activation_scores(score_input, logit_layer)

        if pred_index is None:
            pred_index = int(logits[0].numpy().tolist().index(
                max(logits[0].numpy().tolist())
            ))
        class_score = logits[:, pred_index]

    gradients = tape.gradient(class_score, conv_outputs)
    if gradients is None:
        raise RuntimeError(f"Gradients are None for '{target_layer.name}'.")

    print(f"[gradcam] grad min={float(tf.reduce_min(gradients)):.6f}  "
          f"max={float(tf.reduce_max(gradients)):.6f}  "
          f"mean={float(tf.reduce_mean(tf.abs(gradients))):.6f}")

    weights    = tf.reduce_mean(gradients, axis=(0, 1, 2))
    activation = conv_outputs[0]
    heatmap    = tf.reduce_sum(activation * weights, axis=-1)

    return heatmap.numpy().astype(np.float32)


def overlay_heatmap(heatmap, original_img, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """Blend Grad-CAM heatmap onto the original image."""
    alpha = float(np.clip(alpha, 0.0, 1.0))

    if hasattr(original_img, "convert"):
        original_img = np.array(original_img.convert("RGB"))
    original_img = np.asarray(original_img).copy()
    if original_img.ndim == 2:
        original_img = np.stack([original_img] * 3, axis=-1)
    if original_img.dtype != np.uint8:
        scale = 255.0 if original_img.max() <= 1.0 else 1.0
        original_img = np.clip(original_img * scale, 0, 255).astype(np.uint8)

    heatmap = np.nan_to_num(np.squeeze(heatmap)).astype(np.float32)
    h, w    = original_img.shape[:2]
    heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

    # Robust percentile stretch — always produces visible hotspots
    p_low  = float(np.percentile(heatmap, 50))
    p_high = float(np.percentile(heatmap, 99))
    if p_high - p_low < 1e-8:
        return original_img

    heatmap = np.clip(heatmap, p_low, p_high)
    heatmap = (heatmap - p_low) / (p_high - p_low)

    heatmap_u8     = np.uint8(255 * heatmap)
    heatmap_colour = cv2.applyColorMap(heatmap_u8, colormap)
    heatmap_colour = cv2.cvtColor(heatmap_colour, cv2.COLOR_BGR2RGB)

    effective_alpha = max(alpha, 0.45)
    blended = cv2.addWeighted(
        original_img.astype(np.float32), 1.0 - effective_alpha,
        heatmap_colour.astype(np.float32), effective_alpha,
        0,
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


__all__ = ["find_last_conv_layer", "make_gradcam_heatmap", "overlay_heatmap"]
