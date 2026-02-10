import streamlit as st
from fastai.vision.all import *

# streamlit is a web application to easily do a rapid prototyping

# labeling mechanism / function
def cat_or_dog(file_name):
    if file_name[0].isupper():
        return "CAT"
    else:
        return "DOG"

cat_vs_dog_model = load_learner("cat_vs_dog_model_fastai2_8_4.pkl")

st.markdown("<h1 style='color: yellow;'>Cat or Dog Classifier</h1>", unsafe_allow_html=True)
#st.title("Cat or Dog")
st.text("Created by Gamas Chang")

uploaded_file = st.file_uploader("Choose as image...", type=["jpg", "png", "jpeg"])
if uploaded_file:
    real_img = PILImage.create(uploaded_file)
    resized_img = real_img.resize((224, 224), Image.NEAREST)
    prediction = cat_vs_dog_model.predict(resized_img)
    print(prediction)
    index = int(prediction[1])
    confident_level = prediction[2][index] * 100

    if confident_level > 90:
        label = f"I am {confident_level:.2f} % sure that it is a {prediction[0]}"
    else:
        label = f"WARNING. It am {confident_level:.2f} % sure that it is a {prediction[0]}"
    st.text(label)
    st.image(uploaded_file)