"""
evaluate.py — Evaluate the trained model on the held-out test set.

Outputs
───────
• Classification report (precision / recall / F1 per class)   → reports/
• Confusion matrix heatmap                                      → plots/
• Per-class and macro-average AUC-ROC                          → stdout

Key fix vs original
────────────────────
Images are always loaded in CLASSES order (not os.listdir order) so label
indices are deterministic across different operating systems and filesystems.
"""

import os
import sys
import time

# Force CPU — avoids Metal / CUDA hangs during evaluation
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from src.config import BEST_MODEL_PATH, CLASSES, PLOTS_DIR, REPORTS_DIR, TEST_DIR


# ── Data loading ─────────────────────────────────────────────────────────────

def load_test_data(test_dir: str, img_size: tuple = (224, 224)):
    """
    Load all test images and their integer labels.

    Labels are always assigned by position in CLASSES (the single source of
    truth), never by os.listdir order, which is non-deterministic.

    Returns
    -------
    images : float32 ndarray of shape (N, H, W, 3)  – VGG16-preprocessed
    labels : int32  ndarray of shape (N,)            – class indices
    """
    class_to_idx = {cls: i for i, cls in enumerate(CLASSES)}
    images, labels = [], []

    for cls in CLASSES:
        cls_dir = os.path.join(test_dir, cls)
        if not os.path.isdir(cls_dir):
            print(f"  ⚠️  Directory not found, skipping: {cls_dir}")
            continue

        loaded = 0
        for fname in sorted(os.listdir(cls_dir)):   # sorted → reproducible order
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            img_path = os.path.join(cls_dir, fname)
            try:
                img = image.load_img(img_path, target_size=img_size)
                arr = preprocess_input(image.img_to_array(img))
                images.append(arr)
                labels.append(class_to_idx[cls])
                loaded += 1
            except Exception as exc:
                print(f"    ⚠️  Skipping {fname}: {exc}")

        print(f"  {cls:<14} {loaded:>4} images  (label index = {class_to_idx[cls]})")

    return (
        np.array(images, dtype=np.float32),
        np.array(labels, dtype=np.int32),
    )


# ── Main evaluation ──────────────────────────────────────────────────────────

def evaluate():
    # Load model
    print(f"Loading model from {BEST_MODEL_PATH} …")
    try:
        model = load_model(BEST_MODEL_PATH, compile=False)
        print("✅ Model loaded.\n")
    except Exception as exc:
        print(f"❌ Failed to load model: {exc}")
        sys.exit(1)

    model.summary()

    # Load test data
    print("\nLoading test images …")
    X_test, y_true = load_test_data(TEST_DIR)
    print(f"\nTotal: {len(X_test)} test images across {len(CLASSES)} classes.")

    if len(X_test) == 0:
        print("❌ No images found. Verify TEST_DIR in config.py.")
        sys.exit(1)

    # Batch inference
    BATCH = 32
    n_batches = (len(X_test) + BATCH - 1) // BATCH
    print(f"\nRunning inference on {n_batches} batches …")

    y_pred_prob_parts = []
    for i in range(0, len(X_test), BATCH):
        t0    = time.time()
        preds = model.predict(X_test[i : i + BATCH], verbose=0)
        dt    = time.time() - t0
        print(f"  Batch {i // BATCH + 1:>3}/{n_batches}  ({dt:.2f}s)")
        y_pred_prob_parts.append(preds)

    y_pred_prob = np.concatenate(y_pred_prob_parts, axis=0)
    y_pred      = np.argmax(y_pred_prob, axis=1)

    # ── Classification report ────────────────────────────────────────────────
    print("\n" + "─" * 60)
    report = classification_report(y_true, y_pred, target_names=CLASSES, digits=4)
    print("Classification Report:\n")
    print(report)

    report_path = os.path.join(REPORTS_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved → {report_path}")

    # ── Confusion matrix ─────────────────────────────────────────────────────
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASSES, yticklabels=CLASSES, ax=ax,
    )
    ax.set_title("Confusion Matrix — Test Set")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    fig.tight_layout()

    cm_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    plt.show()
    plt.close(fig)
    print(f"Saved → {cm_path}")

    # ── Per-class AUC-ROC ────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    y_true_bin = label_binarize(y_true, classes=list(range(len(CLASSES))))
    auc_scores = roc_auc_score(y_true_bin, y_pred_prob, average=None)

    print("Per-class AUC-ROC:")
    for cls, auc in zip(CLASSES, auc_scores):
        print(f"  {cls:<14} {auc:.4f}")
    print(f"\n  Macro-average AUC: {np.mean(auc_scores):.4f}")

    print("\n✅ Evaluation complete.")


if __name__ == "__main__":
    evaluate()