"""
train.py — Two-phase training for the brain-tumor VGG16 classifier.

Phase 1  Train only the new classification head (VGG16 base frozen).
Phase 2  Fine-tune the top VGG16 convolutional blocks with a small LR.

Key fixes vs original
─────────────────────
• class_weight passed in Phase 1 (was missing → biased toward majority class)
• Functional-API model from model.py (was Sequential → broke Grad-CAM)
• Callbacks recreated per phase so ModelCheckpoint state does not bleed over
• plot_history closes the figure to prevent matplotlib memory leaks
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.optimizers import Adam

from src.config import (
    BEST_MODEL_PATH,
    EPOCHS,
    FINE_TUNE_LR,
    FINAL_MODEL_PATH,
    LEARNING_RATE,
    PATIENCE,
    PLOTS_DIR,
)
from src.data_loader import get_train_val_generators
from src.model import build_model


# ── Utilities ────────────────────────────────────────────────────────────────

def _make_callbacks():
    """Fresh callback list for each training phase."""
    return [
        EarlyStopping(
            monitor="val_accuracy",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            BEST_MODEL_PATH,
            save_best_only=True,
            monitor="val_accuracy",
            mode="max",
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def plot_history(history, title: str = "Training History", suffix: str = ""):
    """Save and display accuracy + loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title)

    axes[0].plot(history.history["accuracy"],     label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"],     label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, f"training_history{suffix}.png")
    fig.savefig(path, dpi=150)
    plt.show()
    plt.close(fig)   # release memory
    print(f"Plot saved → {path}")


# ── Phase 1: head-only training ──────────────────────────────────────────────

def compile_and_train(
    model,
    train_gen,
    val_gen,
    class_weight_dict: dict,
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
):
    """
    Compile and train the classification head (VGG16 base is frozen).
    class_weight_dict is required — without it minority classes are ignored.
    """
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        class_weight=class_weight_dict,
        callbacks=_make_callbacks(),
    )
    return history


# ── Phase 2: fine-tuning ─────────────────────────────────────────────────────

def fine_tune(
    model,
    base_model,
    train_gen,
    val_gen,
    class_weight_dict: dict,
    epochs: int = 20,
    lr: float = FINE_TUNE_LR,
    unfreeze_from: int = -30,
):
    """
    Unfreeze the last abs(unfreeze_from) layers of base_model and fine-tune
    with a small LR so early VGG features are not destroyed.
    Must recompile after changing trainable flags (Keras requirement).
    """
    base_model.trainable = True
    for layer in base_model.layers[:unfreeze_from]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        class_weight=class_weight_dict,
        callbacks=_make_callbacks(),
    )
    return history


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Data generators
    train_gen, val_gen = get_train_val_generators()

    # 2. Balanced class weights (handles dataset imbalance)
    y_train        = train_gen.classes
    unique_classes = np.unique(y_train)
    weights        = compute_class_weight("balanced", classes=unique_classes, y=y_train)
    class_weight_dict = dict(zip(unique_classes.tolist(), weights.tolist()))
    print("Class weights:", class_weight_dict)

    # 3. Build model (frozen VGG16 base)
    model, base_model = build_model(freeze_base=True)
    model.summary()

    # 4. Phase 1 – train head only
    print("\n─── Phase 1: Training classification head ───")
    history1 = compile_and_train(
        model, train_gen, val_gen,
        class_weight_dict=class_weight_dict,
        epochs=30,
    )
    plot_history(history1, title="Phase 1 – Head Training", suffix="_phase1")

    # 5. Phase 2 – fine-tune top VGG blocks
    print("\n─── Phase 2: Fine-tuning top VGG16 layers ───")
    history2 = fine_tune(
        model, base_model, train_gen, val_gen,
        class_weight_dict=class_weight_dict,
        epochs=20,
    )
    plot_history(history2, title="Phase 2 – Fine-tuning", suffix="_phase2")

    # 6. Save final model
    model.save(FINAL_MODEL_PATH)
    print(f"\n✅ Final model saved → {FINAL_MODEL_PATH}")