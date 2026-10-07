"""
app.py - Plant Disease Detection (Streamlit interface for the trained CNN)
Run with:  streamlit run app.py
"""
import os
import json
import html

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

MODEL_PATH = os.path.join("model", "plant_cnn.keras")
CLASSES_PATH = os.path.join("model", "class_names.json")
IMG_SIZE = (128, 128)

st.set_page_config(page_title="Plant Disease Detection", layout="wide")

# ---------------- Black & white theme (no gradients, no animations) ----------------
st.markdown("""
<style>
.stApp, [data-testid="stHeader"] { background-color: #0a0a0a; color: #ffffff; }
#MainMenu, footer { visibility: hidden; }
.block-container { max-width: 1050px; padding-top: 2.5rem; }
h1.title { font-size: 2rem; font-weight: 700; letter-spacing: 3px; margin: 0; color: #ffffff; }
p.subtitle { color: #a3a3a3; margin: 4px 0 8px 0; letter-spacing: 1px; }
hr.rule { border: 0; border-top: 1px solid #262626; margin: 18px 0 26px 0; }
h3.sec { color: #ffffff; font-size: 0.8rem; letter-spacing: 2px; font-weight: 600; margin: 0 0 12px 0; }
label, [data-testid="stWidgetLabel"] p { color: #e5e5e5 !important; }
[data-testid="stFileUploader"] section { background: #141414; border: 1px dashed #404040; border-radius: 10px; }
[data-testid="stFileUploader"] section * { color: #d4d4d4 !important; }
.stButton > button, [data-testid="stFileUploader"] button {
    background: #ffffff; color: #000000 !important; border: 1px solid #ffffff;
    border-radius: 8px; font-weight: 600; padding: 0.55rem 1.4rem;
}
.stButton > button p, [data-testid="stFileUploader"] button p { color: #000000 !important; }
.stButton > button:hover, [data-testid="stFileUploader"] button:hover {
    background: #d4d4d4; border-color: #d4d4d4; color: #000000 !important;
}
[data-testid="stImage"] img { border: 1px solid #333; border-radius: 8px; }
[data-testid="stImageCaption"], figcaption { color: #a3a3a3 !important; }
.card { background: #141414; border: 1px solid #333; border-radius: 10px; padding: 20px 24px; }
.card .lbl { color: #a3a3a3; font-size: 0.72rem; letter-spacing: 2px; margin-bottom: 4px; }
.card .val { color: #ffffff; font-size: 1.6rem; font-weight: 700; margin-bottom: 18px; }
.card .val:last-child { margin-bottom: 0; }
.card .sub { color: #a3a3a3; font-size: 0.9rem; font-weight: 400; }
.row { display: flex; justify-content: space-between; color: #e5e5e5; margin-bottom: 4px; }
.row span:last-child { color: #ffffff; font-weight: 600; }
.bar { height: 6px; background: #262626; border-radius: 3px; margin-bottom: 16px; }
.bar > div { height: 6px; background: #ffffff; border-radius: 3px; }
.info { color: #a3a3a3; font-size: 0.9rem; line-height: 1.6; }
.msg { background: #141414; border: 1px solid #333; border-radius: 10px; padding: 14px 18px; color: #d4d4d4; }
</style>
""", unsafe_allow_html=True)


def message(text):
    st.markdown(f'<div class="msg">{html.escape(text)}</div>', unsafe_allow_html=True)


# ---------------- Model loading (cached: loaded only once) ----------------
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    try:
        import tensorflow as tf
        return tf.keras.models.load_model(MODEL_PATH), None
    except Exception as e:
        print("Model load error:", repr(e))      # details go to the terminal, not the UI
        return None, "Could not load the model. Check that the TensorFlow/Keras version matches the one used for training."


def load_class_names():
    try:
        with open(CLASSES_PATH, "r", encoding="utf-8") as f:
            names = json.load(f)
        return names if isinstance(names, list) and names else None
    except Exception:
        return None


def predict(model, class_names, image, k=3):
    """RGB -> resize 128x128 -> NumPy -> batch -> model. Returns top-k [(label, confidence %)]."""
    import tensorflow as tf
    arr = tf.image.resize(np.array(image, dtype=np.float32), IMG_SIZE)   # same bilinear resize as training loader
    arr = tf.expand_dims(arr, 0)                                          # (1, 128, 128, 3), raw 0-255
    probs = model(arr, training=False).numpy()[0]                         # softmax output; no augmentation
    top = np.argsort(probs)[::-1][:min(k, len(class_names))]
    return [(class_names[i], float(probs[i]) * 100.0) for i in top]


def pretty(label):
    plant, _, cond = label.partition("___")
    clean = lambda s: s.replace("_", " ").strip()
    plant, cond = clean(plant), clean(cond)
    return plant, cond[:1].upper() + cond[1:]


# ---------------- Header ----------------
st.markdown('<h1 class="title">PLANT DISEASE DETECTION</h1>'
            '<p class="subtitle">CNN-Based Plant Disease Classification</p><hr class="rule">',
            unsafe_allow_html=True)

# ---------------- Checks ----------------
if not os.path.exists(MODEL_PATH):
    message("Trained model not found. Please train the model first.")
    st.stop()
class_names = load_class_names()
if class_names is None:
    message("class_names.json not found or invalid in the model folder.")
    st.stop()
model, err = load_model()
if err:
    message(err)
    st.stop()
if model.output_shape[-1] != len(class_names):
    message("The model output size does not match class_names.json.")
    st.stop()

# ---------------- Main layout ----------------
left, right = st.columns(2, gap="large")
image = None

with left:
    st.markdown('<h3 class="sec">UPLOAD LEAF IMAGE</h3>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"],
                                label_visibility="collapsed")
    if uploaded is not None:
        try:
            image = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
            st.image(image, caption=uploaded.name, use_container_width=True)
        except Exception:
            message("This file could not be read as an image. Please upload a valid JPG or PNG.")
            uploaded = None
        sig = (uploaded.name, uploaded.size) if uploaded is not None else None
        if st.session_state.get("sig") != sig:
            st.session_state["sig"] = sig
            st.session_state.pop("result", None)
    else:
        st.session_state.pop("result", None)
        st.session_state.pop("sig", None)
        message("Upload a leaf image to begin analysis.")

    if image is not None and st.button("Analyze Leaf"):
        try:
            with st.spinner("Analyzing..."):
                st.session_state["result"] = predict(model, class_names, image)
        except Exception as e:
            print("Prediction error:", repr(e))
            message("Prediction failed. Please try another image.")

result = st.session_state.get("result") if image is not None else None

with right:
    st.markdown('<h3 class="sec">PREDICTION RESULT</h3>', unsafe_allow_html=True)
    if result:
        plant, disease = pretty(result[0][0])
        st.markdown(
            f'<div class="card">'
            f'<div class="lbl">PREDICTED CLASS</div>'
            f'<div class="val">{html.escape(disease)} <span class="sub">&nbsp;{html.escape(plant)}</span></div>'
            f'<div class="lbl">CONFIDENCE</div>'
            f'<div class="val">{result[0][1]:.1f}%</div></div>', unsafe_allow_html=True)
    else:
        message("Click \"Analyze Leaf\" to see the prediction." if image is not None
                else "No result yet.")

# ---------------- Top 3 ----------------
if result:
    st.markdown('<hr class="rule"><h3 class="sec">TOP PREDICTIONS</h3>', unsafe_allow_html=True)
    rows = ""
    for label, pct in result:
        plant, disease = pretty(label)
        rows += (f'<div class="row"><span>{html.escape(plant)} - {html.escape(disease)}</span>'
                 f'<span>{pct:.1f}%</span></div>'
                 f'<div class="bar"><div style="width:{min(pct, 100):.1f}%"></div></div>')
    st.markdown(f'<div class="card">{rows}</div>', unsafe_allow_html=True)

# ---------------- Info ----------------
st.markdown('<hr class="rule"><h3 class="sec">ABOUT THIS PREDICTION</h3>', unsafe_allow_html=True)
st.markdown(
    f'<div class="card info">The result is generated by a convolutional neural network (CNN) trained on '
    f'leaf images from {len(class_names)} classes. Confidence is the softmax probability the model assigns '
    f'to the predicted class; it reflects the model\'s certainty, not a guarantee of correctness. '
    f'The model can only recognize the {len(class_names)} classes it was trained on, and works best on a '
    f'single, clearly visible leaf.</div>', unsafe_allow_html=True)