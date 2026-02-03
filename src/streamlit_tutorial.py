import streamlit as st

st.markdown("<h1 style='color: yellow;'>Cat or Dog Classifier</h1>", unsafe_allow_html=True)
#st.title("Cat or Dog")
st.text("Created by Gamas Chang")

uploaded_file = st.file_uploader("Choose as image...", type=["jpg", "png", "jpeg"])
if uploaded_file:
    st.image(uploaded_file)
