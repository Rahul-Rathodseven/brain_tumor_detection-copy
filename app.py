"""
app.py — Streamlit Brain Tumor MRI Classifier.

Run with:
    streamlit run app.py
"""

import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from src.config import BEST_MODEL_PATH, CLASSES
from src.gradcam import make_gradcam_heatmap, overlay_heatmap
from src.predict import load_and_preprocess, predict_image

# ── MUST be the very first Streamlit call ─────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="centered",
)

# ── Display labels: internal name → clean UI label ───────────────────────────
DISPLAY_LABELS = {
    "glioma":      "Glioma",
    "meningioma":  "Meningioma",
    "notumor":     "No Tumor",
    "pituitary":   "Pituitary",
}

def display(cls: str) -> str:
    return DISPLAY_LABELS.get(cls.lower(), cls.capitalize())


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_trained_model():
    if not os.path.exists(BEST_MODEL_PATH):
        st.error(
            f"**Model not found:** `{BEST_MODEL_PATH}`\n\n"
            "Train the model first:\n```\npython -m src.train\n```"
        )
        st.stop()
    model = tf.keras.models.load_model(BEST_MODEL_PATH, compile=False)
    _     = model(tf.zeros((1, 224, 224, 3)), training=False)   # warm-up
    return model


model = load_trained_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Model Info")
    st.markdown("**Internal class → label mapping:**")
    for i, cls in enumerate(CLASSES):
        st.markdown(f"- `{i}` → {display(cls)}")
    st.divider()
    st.caption(
        "Class order is fixed by how the model was trained on Kaggle. "
        "It must match `CLASSES` in `src/config.py`."
    )

# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("🧠 Brain Tumor MRI Classification")
st.markdown(
    "Upload a brain MRI scan to classify it as "
    "**Glioma**, **Meningioma**, **Pituitary tumor**, or **No Tumor**."
)
st.divider()

uploaded_file = st.file_uploader(
    "Choose an MRI image (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:

    pil_image = Image.open(uploaded_file).convert("RGB")
    st.image(pil_image, caption="Uploaded MRI scan", width=300)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        pil_image.save(tmp.name)
        temp_path = tmp.name

    try:
        with st.spinner("Analysing …"):
            predicted_class, confidence, probs = predict_image(model, temp_path)

        # ── Prediction results ────────────────────────────────────────────────
        st.divider()
        col1, col2 = st.columns(2)
        col1.metric("Prediction", display(predicted_class))
        col2.metric("Confidence", f"{confidence:.2%}")

        # FIXED: convert probs to plain Python list to avoid numpy.int64
        # being passed as pred_index into Grad-CAM, which causes StridedSlice
        # "slice index out of bounds" in TF 2.20 / Keras 3.x
        probs_list = probs.tolist() if hasattr(probs, "tolist") else list(probs)
        pred_idx   = int(probs_list.index(max(probs_list)))

        # Probability bar chart
        display_names = [display(c) for c in CLASSES]
        bar_colors    = ["crimson" if c == predicted_class else "steelblue" for c in CLASSES]
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(display_names, probs_list, color=bar_colors, edgecolor="white")
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1.15)
        ax.set_title("Class Probabilities")
        for i, v in enumerate(probs_list):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # ── Grad-CAM ──────────────────────────────────────────────────────────
        st.divider()
        show_gradcam = st.checkbox("🔍 Show Grad-CAM explanation")

        if show_gradcam:
            alpha = st.slider("Heatmap opacity", 0.0, 1.0, 0.4, 0.05)

            try:
                with st.spinner("Generating Grad-CAM …"):
                    preprocessed    = load_and_preprocess(temp_path)
                    # squeeze batch dim: (1,224,224,3) → (224,224,3)
                    img_for_gradcam = np.squeeze(preprocessed, axis=0)

                    # pred_idx is already a plain Python int from probs_list above
                    heatmap = make_gradcam_heatmap(img_for_gradcam, model, pred_idx)
                    overlay = overlay_heatmap(heatmap, np.array(pil_image), alpha=alpha)

                fig2, axes = plt.subplots(1, 2, figsize=(9, 4))
                axes[0].imshow(pil_image)
                axes[0].set_title("Original MRI", fontsize=12)
                axes[0].axis("off")

                axes[1].imshow(overlay)
                axes[1].set_title(
                    f"Grad-CAM  ·  {display(predicted_class)} ({confidence:.1%})",
                    fontsize=12,
                )
                axes[1].axis("off")

                fig2.suptitle(
                    "Highlighted regions show where the model focused its attention",
                    fontsize=10, color="grey",
                )
                fig2.tight_layout()
                st.pyplot(fig2)
                plt.close(fig2)
                st.caption("🔴 Red/yellow = high attention  ·  🔵 Blue = low attention")

            except Exception as exc:
                st.error(f"Grad-CAM failed: {exc}")

        # ── Disclaimer ────────────────────────────────────────────────────────
        st.divider()
        st.warning(
            "⚠️ **Medical Disclaimer:** This tool is for research and educational "
            "purposes only. It is **not** a substitute for professional medical "
            "diagnosis. Always consult a qualified medical professional."
        )

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)