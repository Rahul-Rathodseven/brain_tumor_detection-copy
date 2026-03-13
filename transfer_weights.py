
"""
transfer_weights.py
────────────────────
Migrate weights from the OLD Sequential(VGG16) model to the NEW
Functional-API model WITHOUT retraining from scratch.

Run from the project root:
    python transfer_weights.py

What it does
────────────
1. Loads your existing trained model (old architecture).
2. Builds the new Functional-API model (same layers, different wiring).
3. Transfers weights layer-by-layer by matching shapes.
4. Runs a quick sanity check to confirm predictions are identical.
5. Overwrites BEST_MODEL_PATH with the new architecture + old weights.

After this script you can use Grad-CAM immediately — no retraining needed.
"""

import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from src.config import BEST_MODEL_PATH, CLASSES, NUM_CLASSES
from src.model import build_model


def transfer_weights(old_model_path: str = BEST_MODEL_PATH):
    print("=" * 60)
    print("Brain Tumor VGG16 — Weight Transfer Script")
    print("=" * 60)

    # ── Load old model ───────────────────────────────────────────────────────
    print(f"\n[1/5] Loading old model from: {old_model_path}")
    try:
        old_model = load_model(old_model_path, compile=False)
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        sys.exit(1)

    print(f"  Old model type : {type(old_model).__name__}")
    print(f"  Old model layers: {len(old_model.layers)}")

    # ── Build new model ──────────────────────────────────────────────────────
    print("\n[2/5] Building new Functional-API model …")
    new_model, _ = build_model(freeze_base=False)
    print(f"  New model layers: {len(new_model.layers)}")

    # ── Transfer weights ─────────────────────────────────────────────────────
    print("\n[3/5] Transferring weights …")

    transferred = 0
    skipped     = 0

    # Flatten all layers from both models (handles nested sub-models like VGG16)
    def flat_layers(model):
        result = []
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                result.extend(flat_layers(layer))
            else:
                result.append(layer)
        return result

    old_layers = [l for l in flat_layers(old_model) if l.get_weights()]
    new_layers = [l for l in flat_layers(new_model) if l.get_weights()]

    for old_l, new_l in zip(old_layers, new_layers):
        old_w = old_l.get_weights()
        new_w = new_l.get_weights()

        # Check all weight shapes match
        if all(ow.shape == nw.shape for ow, nw in zip(old_w, new_w)):
            new_l.set_weights(old_w)
            transferred += 1
            print(f"  ✅ {new_l.name:<35} transferred ({len(old_w)} tensors)")
        else:
            skipped += 1
            print(f"  ⚠️  {new_l.name:<35} SKIPPED (shape mismatch)")
            for i, (ow, nw) in enumerate(zip(old_w, new_w)):
                if ow.shape != nw.shape:
                    print(f"       tensor[{i}]: old={ow.shape}  new={nw.shape}")

    print(f"\n  Transferred: {transferred}  |  Skipped: {skipped}")

    if skipped > 0:
        print(
            "\n  ⚠️  Some layers were skipped. This may indicate an architecture "
            "mismatch. Verify that config.py (NUM_CLASSES, etc.) matches the "
            "old model exactly."
        )

    # ── Sanity check ─────────────────────────────────────────────────────────
    print("\n[4/5] Sanity check — comparing predictions …")
    dummy = tf.random.uniform((4, 224, 224, 3))

    old_preds = old_model(dummy, training=False).numpy()
    new_preds = new_model(dummy, training=False).numpy()

    max_diff = float(np.max(np.abs(old_preds - new_preds)))
    print(f"  Max absolute prediction difference: {max_diff:.6f}")

    if max_diff < 1e-4:
        print("  ✅ Predictions match — weight transfer successful.")
    else:
        print(
            "  ⚠️  Predictions differ more than expected. "
            "Check for skipped layers above. You may need to retrain."
        )

    # ── Save ─────────────────────────────────────────────────────────────────
    print(f"\n[5/5] Saving new model to: {old_model_path}")
    new_model.save(old_model_path)
    print("  ✅ Saved.")

    print("\n" + "=" * 60)
    print("Done! Your model now uses the Functional API.")
    print("Grad-CAM will work correctly with app.py.")
    print("=" * 60)


if __name__ == "__main__":
    transfer_weights()