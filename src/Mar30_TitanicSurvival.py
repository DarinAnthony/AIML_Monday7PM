import streamlit as st
from fastai.vision.all import load_learner
import pandas as pd
import numpy as np

st.set_page_config(page_title="Titanic Survival Rate with AI", layout="centered")

@st.cache_resource
def get_model():
    return load_learner("models/titanic_model_fastai_2_8_4.pkl")

titanic_model = get_model()

st.title("Titanic Survival Rate with AI")
st.write("Input your detail below to know if you survive!")

# COLLECTING USER INPUT
name = st.text_input("What is your name: ")
age = st.number_input("How old are you: ", min_value=0, step=1)
gender = st.radio("What is your gender: ", ("male", "female"))
fare = st.number_input("Enter your ticket price: ", min_value=0.0, step=0.01)
# pclass = int(st.radio("What is your priority class: ", (1, 2, 3)))
pclass = 1
sib_sp = 0
parch = 0

if st.button("Submit"):
    new_passenger = {
        "Sex": [gender],
        "Age": [age],
        "Fare": [fare],
        "Pclass": [pclass],
        "SibSp": [sib_sp],
        "Parch": [parch]
    }
    new_df = pd.DataFrame(new_passenger)
    dl = titanic_model.dls.test_dl(new_df)
    preds, targs = titanic_model.get_preds(dl=dl)
    survive_prob = float(preds[0][1])
    st.write("#### THE AI's PREDICTION ####")
    st.markdown(
        f"<p style='color:red'>{name.title()}, the chance you survive the Titanic incident is: {survive_prob:.2f}</p>",
        unsafe_allow_html=True
    )

    # prediction = titanic_model.predict(new_df)
