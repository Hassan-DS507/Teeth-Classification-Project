import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

# Project title
st.title("Teeth Classification")

# Load model (make sure it's in the same folder)
model = load_model("saved_models/best_teeth_model.h5")

# Class names based on training
class_names = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = img.resize((224, 224))
    img_array = keras_image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    pred_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.write(f"Prediction: **{pred_class}**")
    st.write(f"Confidence: {confidence:.2f}%")
