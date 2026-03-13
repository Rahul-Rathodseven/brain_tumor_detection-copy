from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models

from src.config import NUM_CLASSES


def build_model(freeze_base: bool = True):
    """
    Build a VGG16 transfer-learning model using the Keras Functional API.

    Why Functional API (not Sequential)?
    ─────────────────────────────────────
    Sequential(VGG16(...)) hides the inner layer graph from Keras.
    This breaks:
      • model.inputs / model.outputs  → needed for Grad-CAM sub-model
      • layer.output tensor access    → needed for fine-tuning layer checks
      • model.summary() layer tree    → nested model shown as single opaque block

    The Functional API keeps every layer's tensor fully visible, so Grad-CAM,
    fine-tuning, and model inspection all work correctly.

    Returns
    -------
    model      : tf.keras.Model  – full model ready to compile
    base_model : tf.keras.Model  – VGG16 base (needed for fine-tuning phase)
    """
    base_model = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
    )
    base_model.trainable = not freeze_base

    # Build on top of the base model's own input/output tensors
    inputs  = base_model.input                   # (None, 224, 224, 3)
    x       = base_model.output                  # (None, 7, 7, 512)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.Dense(256, activation="relu")(x)
    x       = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name="brain_tumor_vgg16",
    )
    return model, base_model