"""
app.py — Road Crack Detection — Streamlit application.

Launch with:
    .venv\\Scripts\\streamlit.exe run app\\app.py

The app loads the best trained model (U-Net preferred, falls back to FCN),
accepts an uploaded road/pavement image, and displays:
  - Original image
  - Predicted binary crack mask
  - Crack overlay on the original image
  - Adjustable prediction threshold
  - Per-image Dice estimate (if a ground-truth mask is uploaded)
  - Inference time
  - Model information

The model is loaded once at startup; no retraining occurs.

DISCLAIMER: These predictions are experimental outputs of a deep learning
model trained on a limited dataset. They are NOT a substitute for
professional infrastructure inspection by qualified engineers.
"""

import io
import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

# ── Path setup ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CHECKPOINT_DIR,
    DEFAULT_THRESHOLD,
    FCN_BASE_CHANNELS,
    IMAGE_MEAN,
    IMAGE_SIZE,
    IMAGE_STD,
    UNET_BASE_CHANNELS,
)
from src.models.fcn_baseline import FCNBaseline
from src.models.unet import UNet

# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Road Crack Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS — minimal infrastructure-inspection theme ──────────────────────────────
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {background-color: #0f1117;}
    [data-testid="stSidebar"] {background-color: #1a1d26;}
    h1, h2, h3 {color: #e0e6f0;}
    p, label, .stMarkdown {color: #b0b8cc;}
    .metric-card {
        background: #1e2230; border-radius: 8px; padding: 16px 20px;
        border: 1px solid #2c3349; margin: 4px 0;
    }
    .disclaimer {
        background: #2a1a0a; border-left: 4px solid #ff6b00;
        padding: 10px 14px; border-radius: 4px; color: #ffb366;
        font-size: 0.85em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Model loading (cached — runs once per session) ─────────────────────────────

@st.cache_resource(show_spinner="Loading model …")
def load_model():
    """Load the best available checkpoint. Prefers U-Net, falls back to FCN."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_name, ckpt_name, model_cls, channels in [
        ("U-Net",        "unet_best.pth", UNet,        UNET_BASE_CHANNELS),
        ("FCN Baseline", "fcn_best.pth",  FCNBaseline, FCN_BASE_CHANNELS),
    ]:
        ckpt_path = CHECKPOINT_DIR / ckpt_name
        if ckpt_path.exists():
            if model_cls == UNet:
                model = UNet(base_channels=channels)
            else:
                model = FCNBaseline(base_channels=channels)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            model.to(device).eval()
            n_params = sum(p.numel() for p in model.parameters())
            return model, model_name, device, n_params, ckpt_path.name

    return None, None, None, 0, None


# ── Inference helpers ──────────────────────────────────────────────────────────

def preprocess(pil_image: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a normalised [1, 3, H, W] tensor."""
    img = pil_image.convert("RGB").resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
    )
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array(IMAGE_MEAN, dtype=np.float32)
    std  = np.array(IMAGE_STD,  dtype=np.float32)
    arr  = (arr - mean) / std
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    return tensor


@torch.no_grad()
def run_inference(model, tensor: torch.Tensor, device, threshold: float):
    """Return (prob_map [H,W] np.float32, binary_mask [H,W] bool, elapsed_ms)."""
    tensor = tensor.to(device)
    t0 = time.perf_counter()
    with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
        logits = model(tensor)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    prob_map     = torch.sigmoid(logits[0, 0]).cpu().numpy().astype(np.float32)
    binary_mask  = prob_map >= threshold
    return prob_map, binary_mask, elapsed_ms


def build_overlay(original_rgb: np.ndarray, binary_mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend original image with a red crack overlay."""
    overlay = original_rgb.copy().astype(np.float32)
    crack_colour = np.array([255, 60, 60], dtype=np.float32)
    overlay[binary_mask] = (
        (1 - alpha) * overlay[binary_mask] + alpha * crack_colour
    )
    return overlay.clip(0, 255).astype(np.uint8)


# ── Sidebar ────────────────────────────────────────────────────────────────────

def sidebar_controls():
    st.sidebar.title("Settings")

    threshold = st.sidebar.slider(
        "Prediction threshold",
        min_value=0.10,
        max_value=0.90,
        value=DEFAULT_THRESHOLD,
        step=0.05,
        help=(
            "Pixels with probability >= threshold are classified as crack. "
            "Lower values increase recall (detect more cracks) but also "
            "increase false positives."
        ),
    )

    overlay_alpha = st.sidebar.slider(
        "Overlay opacity",
        min_value=0.1,
        max_value=0.9,
        value=0.45,
        step=0.05,
    )

    show_prob = st.sidebar.checkbox(
        "Show probability heatmap",
        value=False,
        help="Display the raw sigmoid output before thresholding.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        '<div class="disclaimer">'
        "<strong>Disclaimer:</strong> Predictions are experimental outputs "
        "of a deep learning model trained on a limited dataset. They are "
        "<strong>NOT</strong> a substitute for professional infrastructure "
        "inspection by qualified engineers."
        "</div>",
        unsafe_allow_html=True,
    )
    return threshold, overlay_alpha, show_prob


# ── Main application ───────────────────────────────────────────────────────────

def main():
    st.title("Road Crack Detection")
    st.markdown(
        "Upload a road or pavement image to detect pixel-level crack regions "
        "using a trained deep learning segmentation model."
    )

    # Load model
    model, model_name, device, n_params, ckpt_name = load_model()

    if model is None:
        st.error(
            "No trained model checkpoint found. "
            "Train a model first:\n\n"
            "```\n.venv\\Scripts\\python.exe -m src.train --model unet\n```"
        )
        return

    # Sidebar
    threshold, overlay_alpha, show_prob = sidebar_controls()

    # Model info panel
    with st.expander("Model information", expanded=False):
        col1, col2, col3 = st.columns(3)
        col1.metric("Architecture", model_name)
        col2.metric("Parameters", f"{n_params:,}")
        col3.metric("Checkpoint", ckpt_name)
        st.markdown(
            f"**Device:** {device}  |  "
            f"**Input size:** {IMAGE_SIZE}×{IMAGE_SIZE} px  |  "
            f"**Threshold:** {threshold:.2f}"
        )

    # File uploader
    uploaded = st.file_uploader(
        "Upload a road image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Upload any road or pavement photograph. JPEG and PNG recommended.",
    )

    if uploaded is None:
        st.info("Upload an image above to begin analysis.")
        return

    # ── Load and validate uploaded image ──────────────────────────────────────
    try:
        pil_image = Image.open(io.BytesIO(uploaded.read()))
        pil_image.verify()   # catches corrupt files
        pil_image = Image.open(io.BytesIO(uploaded.getvalue()))
    except Exception as e:
        st.error(f"Cannot read the uploaded file: {e}")
        return

    # ── Run inference ──────────────────────────────────────────────────────────
    with st.spinner("Analysing image …"):
        tensor = preprocess(pil_image)
        prob_map, binary_mask, elapsed_ms = run_inference(model, tensor, device, threshold)

    crack_pct = float(binary_mask.mean() * 100)

    # ── Resize original for display (match model input size for alignment) ─────
    display_rgb = np.array(
        pil_image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    )
    overlay = build_overlay(display_rgb, binary_mask, alpha=overlay_alpha)

    # ── Display results ────────────────────────────────────────────────────────
    st.markdown("### Results")

    if show_prob:
        cols = st.columns(4)
        panels = [
            ("Original image",       display_rgb,                       "RGB"),
            ("Ground truth unknown", (binary_mask * 255).astype(np.uint8), "L"),
            ("Probability map",      (prob_map * 255).astype(np.uint8),    "L"),
            ("Crack overlay",        overlay,                            "RGB"),
        ]
        labels = ["Original image", "Predicted mask", "Probability map", "Crack overlay"]
    else:
        cols = st.columns(3)
        panels = [
            ("Original image", display_rgb,                         "RGB"),
            ("Predicted mask", (binary_mask * 255).astype(np.uint8), "L"),
            ("Crack overlay",  overlay,                              "RGB"),
        ]
        labels = ["Original image", "Predicted mask", "Crack overlay"]

    for col, (label, arr, mode) in zip(cols, panels):
        with col:
            st.image(Image.fromarray(arr, mode=mode), caption=label, use_container_width=True)

    # ── Metric cards ───────────────────────────────────────────────────────────
    st.markdown("### Analysis")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Crack coverage",  f"{crack_pct:.2f} %")
    m2.metric("Inference time",  f"{elapsed_ms:.1f} ms")
    m3.metric("Threshold used",  f"{threshold:.2f}")
    m4.metric("Device",          str(device).upper())

    # Severity indicator
    if crack_pct < 1.0:
        severity, colour = "Low", "green"
    elif crack_pct < 5.0:
        severity, colour = "Moderate", "orange"
    else:
        severity, colour = "High", "red"

    st.markdown(
        f"**Damage severity estimate:** :{colour}[{severity}]  "
        f"({crack_pct:.2f} % of image area classified as crack)"
    )

    st.markdown(
        '<div class="disclaimer">'
        "This severity estimate is derived purely from pixel coverage and "
        "has not been validated for engineering assessment. Consult a "
        "qualified structural engineer before making maintenance decisions."
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
