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
import io

st.set_page_config(page_title="Jellyfish Classifier", layout="centered")

def extract_breed(file_name):
    # you remove the extension of a filename using these next two lines of code
    # and it specifically gives you a list of strings that were separated by _
    path_obj = Path(file_name)
    parts = path_obj.stem.split('_')

    # print(parts)

    breed = '_'.join(parts[:-1]) # group all the elements in 'parts' except the last one
    # ^ this groups each element with an _

    if file_name[0].isupper():
        return "CAT - " + breed
    else:
        return "DOG - "+ breed

@st.cache_resource
def get_model():
    return load_learner("models/jellyfish_model_fastai_2_7_19.pkl")

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