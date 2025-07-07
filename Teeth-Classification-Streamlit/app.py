import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import json
from streamlit_lottie import st_lottie
import requests

# ======================
# Configuration
# ======================
st.set_page_config(
    page_title="Teeth Classification AI",
    layout="centered",
    page_icon="🦷",
    initial_sidebar_state="collapsed"
)

# ======================
# Constants
# ======================
CLASS_NAMES = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']

# Lottie animation URL fallback
LOTTIE_URL = "https://lottie.host/8f03bc4c-a498-4655-b265-2fdc3c4eeb64/NJxbmu83fg.json"

# ======================
# Disease Information
# ======================
DISEASE_INFO = {
    'CaS': {
        'name': "Caries Superficial",
        'description': "Early stage of tooth decay affecting the outer enamel layer.",
        'recommendation': "Recommend fluoride treatment and improved oral hygiene. Schedule a follow-up in 3 months."
    },
    'CoS': {
        'name': "Composite Superficial",
        'description': "Shallow dental filling material visible on the tooth surface.",
        'recommendation': "No immediate treatment needed. Routine monitoring recommended."
    },
    'Gum': {
        'name': "Gum Area",
        'description': "Image focuses on gum tissue, showing potential periodontal concerns.",
        'recommendation': "Consider periodontal evaluation. Recommend dental cleaning and gum health assessment."
    },
    'MC': {
        'name': "Metal Crown",
        'description': "Artificial metal restoration covering a damaged tooth.",
        'recommendation': "Routine crown monitoring recommended. Check for margins and integrity."
    },
    'OC': {
        'name': "Orthodontic Component",
        'description': "Braces or other alignment devices visible in the image.",
        'recommendation': "Regular orthodontic follow-up recommended. Monitor tooth movement."
    },
    'OLP': {
        'name': "Oral Lichen Planus",
        'description': "Chronic inflammatory condition affecting oral mucous membranes.",
        'recommendation': "Refer to oral medicine specialist. May require biopsy for confirmation."
    },
    'OT': {
        'name': "Other",
        'description': "Condition not matching standard categories. Requires specialist review.",
        'recommendation': "Recommend consultation with oral radiologist or specialist."
    }
}

# ======================
# Custom Styles
# ======================
def load_css():
    st.markdown("""
    <style>
        /* Main container */
        .main {
            max-width: 800px;
            padding: 2rem;
            margin: 0 auto;
        }
        
        /* Header section */
        .header {
            text-align: center;
            margin-bottom: 2rem;
        }
        
        /* Upload section */
        .upload-container {
            background: #f8f9fa;
            border-radius: 12px;
            padding: 2rem;
            margin: 2rem auto;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            max-width: 700px;
        }
        
        /* Result cards */
        .result-card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem auto;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            border-left: 4px solid #4CAF50;
            max-width: 700px;
        }
        
        .diagnosis-card {
            background: #e3f2fd;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem auto;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            max-width: 700px;
        }
        
        .recommendation-card {
            background: #fff8e1;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem auto;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            border-left: 4px solid #ffc107;
            max-width: 700px;
        }
        
        /* Progress bar */
        .stProgress > div > div > div {
            background-color: #4CAF50;
        }
        
        /* Hide sidebar */
        section[data-testid="stSidebar"] {
            display: none !important;
        }
        
        /* Animation */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .fade-in {
            animation: fadeIn 0.5s ease-out;
        }
        
        /* Center content */
        .stApp {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
    </style>
    """, unsafe_allow_html=True)

# ======================
# Helper Functions
# ======================
def load_lottie_animation():
    """Load Lottie animation with fallback to URL"""
    try:
        # First try local file
        with open("animation.json", "r") as f:
            return json.load(f)
    except:
        try:
            # Fallback to URL if local file not found
            response = requests.get(LOTTIE_URL)
            if response.status_code == 200:
                return response.json()
        except:
            return None
    return None

def load_model_safely():
    """Load model with error handling"""
    try:
        return load_model("../saved_models/best_teeth_model.h5")
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()

def preprocess_image(img):
    """Preprocess image for model prediction"""
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# ======================
# App Layout
# ======================
def main():
    load_css()
    
    # Header Section
    st.markdown("""
    <div class="header fade-in">
        <h1 style="color:#2e86de;">Teeth Classification</h1>
        <h3 style="color:#555;">AI for Dental Diagnostics</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Load and display animation
    lottie_anim = load_lottie_animation()
    if lottie_anim:
        st_lottie(lottie_anim, height=200, key="header-anim")
    else:
        # Fallback static image
        st.image("https://images.unsplash.com/photo-1588776814546-1ffcf47267a5?ixlib=rb-1.2.1&auto=format&fit=crop&w=800&q=80", 
                use_container_width=True)
    
    # Project Overview
    st.markdown("""
<h4 style="color:#2e86de; margin-top:1.5rem;">Production Pipeline</h4>
<ul>
    <li>Image preprocessing (normalization, augmentation)</li>
    <li>Visual inspection and quality control</li>
    <li>Deep learning model training with TensorFlow</li>
    <li>Real-time prediction capabilities</li>
</ul>
""", unsafe_allow_html=True)

    
    st.markdown("---")
    
    # Image Upload Section
    st.markdown("""
    <div class="fade-in">
        <h2 style="color:#2e86de;">Diagnostic Tool</h2>
        <p>Upload a dental image for AI classification</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a dental X-ray image", 
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption="Uploaded Image", use_container_width=True)
        
        # Load model and predict
        try:
            model = load_model_safely()
            img_array = preprocess_image(img)
            
            with st.spinner("Analyzing dental image..."):
                prediction = model.predict(img_array)
                pred_index = np.argmax(prediction)
                pred_class = CLASS_NAMES[pred_index]
                confidence = float(np.max(prediction)) * 100
                
                # Get disease info
                disease = DISEASE_INFO.get(pred_class, {})
                
            # Display results
            st.markdown(f"""
            <div class="result-card fade-in">
                <h3 style="color:#2e86de; margin-bottom:0.5rem;">Diagnosis Result</h3>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h2 style="margin:0; color:#222;">{pred_class}</h2>
                        <p style="margin:0; font-size:1rem; color:#555;">{disease.get('name', '')}</p>
                    </div>
                    <div style="text-align: right;">
                        <p style="margin:0; font-size:1rem; color:#555;">Confidence</p>
                        <h3 style="margin:0; color:#4CAF50;">{confidence:.1f}%</h3>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.progress(int(confidence))
            
            # Diagnosis information
            st.markdown(f"""
            <div class="diagnosis-card fade-in">
                <h4 style="color:#2e86de; margin-top:0;">Condition Information</h4>
                <p>{disease.get('description', 'No information available')}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Recommendations
            st.markdown(f"""
            <div class="recommendation-card fade-in">
                <h4 style="color:#ff9800; margin-top:0;">Clinical Recommendations</h4>
                <p>{disease.get('recommendation', 'Consult with dental specialist')}</p>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            
    else:
        st.markdown("""
        <div class="upload-container fade-in">
            <div style="text-align:center; padding:2rem;">
                <h4 style="color:#555;"> Upload a image to begin analysis</h4>
                <p style="color:#777;">Supported formats: JPG, JPEG, PNG</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#666; margin-top:2rem;">
        <p>© 2025 Hassan Abdelrazek | Powered by TensorFlow</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()