"""Streamlit decision-support demo for brain-tumor-ssl.

Upload a brain-MRI slice, get the model's 4-class prediction with per-class
probabilities, and an attention-rollout heatmap showing which regions drove the
decision. This is a thin UI over ``brain_tumor_ssl.inference``; all real work lives
in the package so it stays testable.

Run:  uv run streamlit run app/streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from brain_tumor_ssl.config import load_config
from brain_tumor_ssl.inference import load_classifier, predict_image

st.set_page_config(page_title="Brain Tumor SSL", page_icon="🧠", layout="wide")

st.title("🧠 Brain Tumor MRI Classifier")
st.warning(
    "**Decision-support research only — NOT a medical device.** Predictions are "
    "unverified, may be wrong, and must not be used for diagnosis or any clinical "
    "decision-making.",
    icon="⚠️",
)


@st.cache_resource(show_spinner="Loading model…")
def _load(config_path: str, checkpoint_path: str, device: str):
    """Load and cache the config + classifier for a given checkpoint/device."""
    cfg = load_config(config_path, overrides={"experiment": {"device": device}})
    clf = load_classifier(Path(checkpoint_path), cfg)
    return cfg, clf


with st.sidebar:
    st.header("Model")
    config_path = st.text_input("Config path", value="configs/config.yaml")
    checkpoint_path = st.text_input("Checkpoint path", value="results/checkpoints/clf.pt")
    device = st.selectbox("Device", options=["cpu", "cuda", "auto"], index=0)
    st.header("Explanation")
    show_explain = st.toggle("Attention overlay", value=True)
    alpha = st.slider("Overlay opacity", 0.0, 1.0, 0.5, 0.05, disabled=not show_explain)
    colormap = st.selectbox("Colormap", ["jet", "inferno", "viridis", "magma"], index=0)

uploaded = st.file_uploader(
    "Upload an MRI slice", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"]
)

if uploaded is None:
    st.info("Upload a grayscale brain-MRI image to get a prediction.")
    st.stop()

if not Path(checkpoint_path).is_file():
    st.error(
        f"Checkpoint not found: `{checkpoint_path}`. Train one first, e.g. "
        "`uv run btssl finetune --output results/checkpoints/clf.pt`, or point the "
        "sidebar at an existing checkpoint."
    )
    st.stop()

try:
    cfg, clf = _load(config_path, checkpoint_path, device)
except Exception as exc:  # surface any load error to the user
    st.exception(exc)
    st.stop()

image = Image.open(uploaded)
prediction = predict_image(
    clf, image, cfg, explain=show_explain, alpha=alpha, colormap=colormap
)

left, right = st.columns(2)
with left:
    st.subheader("Input")
    st.image(image, caption=uploaded.name, use_container_width=True)
with right:
    st.subheader("Attention overlay" if prediction.overlay is not None else "Prediction")
    if prediction.overlay is not None:
        st.image(
            prediction.overlay,
            caption="Attention rollout (warmer = more influential)",
            use_container_width=True,
        )

top_p = prediction.probabilities[prediction.label]
st.metric("Prediction", prediction.label, delta=f"{top_p:.1%} confidence")

probs = pd.DataFrame(
    {
        "class": list(prediction.probabilities),
        "probability": list(prediction.probabilities.values()),
    }
).set_index("class")
st.bar_chart(probs, horizontal=True)
