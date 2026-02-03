from fastai.vision.all import *
import streamlit as st
from pathlib import Path
from PIL import Image as PILImageLib  # only for Image.NEAREST

st.markdown("<h1 style='color: yellow;'>Cat or Dog Classifier</h1>", unsafe_allow_html=True)
st.text("Created by Darin Djapri")

MODELS_DIR = Path("models")

# Find all .pkl files in models/
model_paths = sorted(MODELS_DIR.glob("*.pkl"))
model_names = [p.name for p in model_paths]

if not model_paths:
    st.error(f"No .pkl models found in: {MODELS_DIR.resolve()}")
    st.stop()

# UI: choose which model to use
selected_name = st.selectbox("Select a model to use:", model_names)

@st.cache_resource  # cache the loaded learner per selected model
def get_model(model_path_str: str):
    return load_learner(model_path_str)

model_path = str(MODELS_DIR / selected_name)
cat_vs_dog_model = get_model(model_path)

st.caption(f"Loaded model: {selected_name}")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file:
    real_img = PILImage.create(uploaded_file)
    resized_img = real_img.resize((224, 224), resample=PILImageLib.NEAREST)

    prediction = cat_vs_dog_model.predict(resized_img)
    index = int(prediction[1])
    confidence_level = float(prediction[2][index]) * 100

    if confidence_level > 90:
        label = f"It is a {prediction[0]} with {confidence_level:.2f}% confidence."
    else:
        label = f"WARNING. Not sure — predicted {prediction[0]} with {confidence_level:.2f}% confidence."

    st.text(label)
    st.image(uploaded_file)