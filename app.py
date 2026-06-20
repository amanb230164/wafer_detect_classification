
import streamlit as st
from PIL import Image
from predict import predict_wafer

st.set_page_config(
    page_title="Wafer Defect Classification",
    layout="centered"
)

st.title("Wafer Defect Classification")

st.write(
    "Upload a wafer map image and the model will predict the defect type."
)

uploaded_file = st.file_uploader(
    "Choose a wafer image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Wafer Image",
        use_container_width=True
    )

    with st.spinner("Predicting..."):
        prediction = predict_wafer(image)

    st.success(
        f"Predicted Defect Type: {prediction}"
    )
