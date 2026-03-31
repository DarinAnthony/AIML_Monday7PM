import streamlit as st
from fastai.vision.all import load_learner, PILImage
from PIL import Image
import io


"""
import numpy
import scipy
import matplotlib
import torch
import fastai
import PIL

print("numpy:", numpy.__version__)
print("scipy:", scipy.__version__)
print("matplotlib:", matplotlib.__version__)
print("torch:", torch.__version__)
print("fastai:", fastai.__version__)
print("pillow (PIL):", PIL.__version__)

OUTPUT:
numpy: 2.0.2
scipy: 1.15.3
matplotlib: 3.10.0
torch: 2.8.0
fastai: 2.8.4
pillow (PIL): 11.3.0
"""


# Must exist when unpickling the Learner
def cat_or_dog(file_name):
    # In training, file_name was something like "Siamese_20.jpg"
    # At inference, it may be a Path. Convert to just the base name.
    name = getattr(file_name, "name", str(file_name)).split("/")[-1]
    return "CAT" if name[0].isupper() else "DOG"

st.set_page_config(page_title="Cat vs Dog Classifier", layout="centered")

@st.cache_resource
def get_model():
    return load_learner("models/cat_vs_dog_model_fastai_2_8_4_rn50.pkl")

learn = get_model()

st.title("Cat vs Dog Classifier")
st.write("Upload an image and I’ll predict whether it’s a CAT or DOG.")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img_bytes = uploaded.read()
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    st.image(pil_img, caption="Uploaded image", use_container_width=True)

    fastai_img = PILImage.create(pil_img)
    pred_class, pred_idx, probs = learn.predict(fastai_img)

    conf = float(probs[int(pred_idx)]) * 100.0
    st.subheader("Prediction")
    st.write(f"**{pred_class}**")
    st.write(f"Confidence: **{conf:.2f}%**")