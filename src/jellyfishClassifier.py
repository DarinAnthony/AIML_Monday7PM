# streamlit run <filename>
# example:
# streamlit run src/Apr1_CatVSDog.py

"""
if you want to deploy this python code on a website online and not locally, make sure you have done
the following in the terminal:

git add .
git commit -m "an update"
git push origin main
"""

import streamlit as st
from fastai.vision.all import load_learner, PILImage
from fastai.vision.all import *
from PIL import Image
import PIL
import io
import fastai
import os

# print("numpy:", numpy.__version__)
# print("scipy:", scipy.__version__)
print("matplotlib:", matplotlib.__version__)
print("torch:", torch.__version__)
print("fastai:", fastai.__version__)
print("pillow (PIL):", PIL.__version__)

st.set_page_config(page_title="Jellyfish Classifier", layout="centered")

def extract_jellyfish(file_path):
    dirname = os.path.dirname(file_path)
    subdirname = os.path.basename(dirname)

    return subdirname

@st.cache_resource
def get_model():
    return load_learner("models/jellyfish_classifier_fastai_2_7_19.pkl")

learn = get_model()

st.title("Jellyfish Classifier")
st.write("Upload an image and I’ll predict the jellyfish breed.")

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