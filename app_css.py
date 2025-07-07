# ===============================
# Teeth Classification Web App (Single Page)
# ===============================

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import json
from streamlit_lottie import st_lottie  

# -------------------------------
# Load Lottie Animation (local)
# -------------------------------
def load_lottie_file(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

try:
    lottie_animation = load_lottie_file("animation.json")
except:
    # Fallback to a simple loading animation from Streamlit
    lottie_animation = None

# -------------------------------
# Load Trained Model
# -------------------------------
model = load_model("saved_models/best_teeth_model.h5")   

# -------------------------------
# Class Labels
# -------------------------------
class_names = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Teeth Classifier",
    layout="centered",
    page_icon="🦷"
)

# Custom CSS for styling
st.markdown("""
<style>
    .centered {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        text-align: center;
    }
    .result-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .upload-box {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
    }
    .stProgress > div > div > div {
        background-color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Main App
# -------------------------------
st.title(" AI Dental Classifier")
st.markdown("Upload a dental X-ray image to predict the tooth class or condition.")

# Animation
if lottie_animation:
    st_lottie(lottie_animation, speed=1, height=200, key="loading")
else:
    st.image("https://cdn.dribbble.com/users/1186261/screenshots/3718681/_______.gif", 
             width=200)

st.markdown("---")

# -------------------------------
# Image Uploader (Main Area)
# -------------------------------
uploaded_file = st.file_uploader(
    "Choose a dental X-ray image", 
    type=["jpg", "jpeg", "png"],
    label_visibility="hidden"
)

if uploaded_file is not None:
    # Display uploaded image
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocessing
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    with st.spinner("Analyzing the image..."):
        prediction = model.predict(img_array)
        pred_index = np.argmax(prediction)
        pred_class = class_names[pred_index]
        confidence = float(np.max(prediction)) * 100

    # Display result
    st.markdown("---")
    
    st.markdown(f"""
    <div class="result-card centered">
        <h3 style="color:#2e86de;">Prediction Result</h3>
        <h2 style="color:#222;">{pred_class}</h2>
        <p style="font-size:18px;color:#555;">Confidence: {confidence:.2f}%</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.progress(int(confidence), text="Confidence Level")
    
    # Optional: Show all probabilities
    with st.expander("See detailed probabilities"):
        for i, (class_name, prob) in enumerate(zip(class_names, prediction[0])):
            st.write(f"{class_name}: {prob*100:.2f}%")

else:
    st.markdown("""
    <div class="upload-box centered">
        <h3 style="color:#555;"> Upload a dental image to begin analysis</h3>
        <p style="color:#777;">Supported formats: JPG, JPEG, PNG</p>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("""
<div style="text-align:center;">
    Developed with  by Hassan | Built with TensorFlow and Streamlit
</div>
""", unsafe_allow_html=True)