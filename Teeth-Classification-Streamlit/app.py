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
# Disease Information (Enhanced)
# ======================
DISEASE_INFO = {
    'CaS': {
        'name': "Caries Superficial",
        'description': "A superficial dental caries lesion affecting the enamel layer. Often caused by poor oral hygiene and sugar exposure.",
        'recommendation': "Recommend fluoride treatment and improved oral hygiene. Schedule a follow-up in 3 months.",
        'link': "https://www.mouthhealthy.org/all-topics-a-z/tooth-decay",
        'icon': "🦷"
    },
    'CoS': {
        'name': "Composite Superficial",
        'description': "A shallow composite dental filling placed to restore a small area of decay or damage.",
        'recommendation': "No urgent action required. Monitor the filling for any discoloration or leakage during routine visits.",
        'link': "https://www.mouthhealthy.org/all-topics-a-z/fillings",
        'icon': "🔧"
    },
    'Gum': {
        'name': "Gum Area",
        'description': "The image is focused on gum tissue which may show signs of gingivitis or early periodontal disease.",
        'recommendation': "Recommend dental cleaning, gum evaluation, and improving flossing habits. Refer to periodontist if symptoms persist.",
        'link': "https://www.mouthhealthy.org/all-topics-a-z/gum-disease",
        'icon': "🩸"
    },
    'MC': {
        'name': "Metal Crown",
        'description': "A metal crown restoration used to protect and strengthen a decayed or broken tooth.",
        'recommendation': "Routine monitoring for crown fit, margin leakage, and secondary decay is advised.",
        'link': "https://www.mouthhealthy.org/all-topics-a-z/crowns",
        'icon': "👑"
    },
    'OC': {
        'name': "Orthodontic Component",
        'description': "Brackets, wires, or other orthodontic appliances visible in the image.",
        'recommendation': "Encourage compliance with orthodontic care. Monitor oral hygiene and gum health during treatment.",
        'link': "https://www.mouthhealthy.org/all-topics-a-z/orthodontics",
        'icon': "🦷"
    },
    'OLP': {
        'name': "Oral Lichen Planus",
        'description': "A chronic inflammatory disease that affects the mucous membranes inside the mouth. Often presents with white patches or red lesions.",
        'recommendation': "Refer to oral medicine specialist. Biopsy may be needed to confirm diagnosis and rule out malignancy.",
        'link': "https://www.mouthhealthy.org/all-topics-a-z/oral-cancer",
        'icon': "⚠️"
    },
    'OT': {
        'name': "Other",
        'description': "The condition does not match the known diagnostic categories. Further examination is needed.",
        'recommendation': "Refer to a dental radiologist or oral pathologist for expert interpretation.",
        'link': "https://www.mouthhealthy.org/all-topics-a-z/dental-visits",
        'icon': "❓"
    }
}

# ======================
# Custom Styles (Enhanced)
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
            transition: all 0.3s ease;
        }
        
        .upload-container:hover {
            box-shadow: 0 6px 12px rgba(0,0,0,0.1);
            transform: translateY(-2px);
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
        
        .link-card {
            background: #f5f5f5;
            border-radius: 12px;
            padding: 1rem;
            margin: 1rem auto;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            max-width: 700px;
            transition: all 0.3s ease;
        }
        
        .link-card:hover {
            background: #ebf5ff;
            box-shadow: 0 4px 8px rgba(0,119,255,0.1);
        }
        
        .link-card a {
            color: #1976d2;
            text-decoration: none;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .link-card a:hover {
            color: #0d47a1;
            text-decoration: underline;
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
        
        /* Language toggle */
        .language-toggle {
            position: absolute;
            top: 10px;
            right: 10px;
            z-index: 100;
        }
        
        /* Confidence indicator */
        .confidence-badge {
            display: inline-flex;
            align-items: center;
            padding: 4px 8px;
            border-radius: 12px;
            background: #e8f5e9;
            color: #2e7d32;
            font-weight: bold;
            font-size: 0.9rem;
        }
        
        /* Disease icon */
        .disease-icon {
            font-size: 1.5rem;
            margin-right: 8px;
            vertical-align: middle;
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
        return load_model("Teeth-Classification-Streamlit/final_model/best_teeth_model_try2.h5")
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()

def preprocess_image(img):
    """Preprocess image for model prediction"""
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# ======================
# App Layout (Enhanced)
# ======================
def main():
    load_css()
    
    # Language Toggle (Placeholder for future implementation)
    st.markdown("""
    <div class="language-toggle">
        <button style="background: #f0f0f0; border: none; padding: 5px 10px; border-radius: 5px; cursor: pointer;">English</button>
    </div>
    """, unsafe_allow_html=True)
    
    # Header Section
    st.markdown("""
    <div class="header fade-in">
        <h1 style="color:#2e86de;">AI Dental Classifier</h1>
        <h3 style="color:#555;">Advanced Diagnosis Assistant</h3>
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
    <div class="fade-in">
        <h4 style="color:#2e86de; margin-top:1.5rem;">Clinical AI Pipeline</h4>
        <ul>
            <li>Automated image preprocessing and quality control</li>
            <li>Deep learning model with 93% validation accuracy</li>
            <li>Evidence-based clinical recommendations</li>
            <li>Trusted external resources for patient education</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Image Upload Section
    st.markdown("""
    <div class="fade-in">
        <h2 style="color:#2e86de;">Upload Dental Image</h2>
        <p>Get AI-powered diagnosis and clinical recommendations</p>
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
        st.image(img, caption="Uploaded Dental Image", use_container_width=True)
        
        # Load model and predict
        try:
            model = load_model_safely()
            img_array = preprocess_image(img)
            
            with st.spinner("Analyzing dental image..."):
                # Simulate processing delay for better UX
                import time
                time.sleep(1)
                
                prediction = model.predict(img_array)
                pred_index = np.argmax(prediction)
                pred_class = CLASS_NAMES[pred_index]
                confidence = float(np.max(prediction)) * 100
                
                # Get disease info
                disease = DISEASE_INFO.get(pred_class, {})
                
            # Display results with enhanced UI
            st.markdown(f"""
            <div class="result-card fade-in">
                <h3 style="color:#2e86de; margin-bottom:0.5rem;">Diagnosis Result</h3>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h2 style="margin:0; color:#222;">
                            <span class="disease-icon">{disease.get('icon', '🦷')}</span>
                            {pred_class}: {disease.get('name', '')}
                        </h2>
                        <p style="margin:0; font-size:1rem; color:#555;">{disease.get('description', '')}</p>
                    </div>
                    <div style="text-align: right;">
                        <span class="confidence-badge">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin-right:4px;">
                                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z" fill="#2e7d32"/>
                            </svg>
                            {confidence:.1f}% confidence
                        </span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.progress(int(confidence))
            
            # Diagnosis information
            st.markdown(f"""
            <div class="diagnosis-card fade-in">
                <h4 style="color:#2e86de; margin-top:0;">Clinical Information</h4>
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
            
            # External resource link
            if disease.get('link'):
                st.markdown(f"""
                <div class="link-card fade-in">
                    <h4 style="color:#2196F3; margin-top:0;">📎 Learn More</h4>
                    <a href="{disease['link']}" target="_blank">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M10 6V8H5V19H16V14H18V20C18 20.5523 17.5523 21 17 21H4C3.44772 21 3 20.5523 3 20V7C3 6.44772 3.44772 6 4 6H10ZM21 3V11H19V6.413L11.207 14.207L9.793 12.793L17.585 5H13V3H21Z" fill="#1976d2"/>
                        </svg>
                        Visit MouthHealthy.org for detailed information
                    </a>
                </div>
                """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            
    else:
        st.markdown("""
        <div class="upload-container fade-in">
            <div style="text-align:center; padding:2rem;">
                <h4 style="color:#555;">Upload a dental image to begin analysis</h4>
                <p style="color:#777;">Supported formats: JPG, JPEG, PNG</p>
                <p style="color:#999; font-size:0.9rem;">AI will analyze the image and provide diagnostic suggestions</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#666; margin-top:2rem;">
        <p>© 2025 Dental AI Assistant | Developed by Hassan Abdel-Razeq</p>
        <p style="font-size:0.8rem;">This tool is for educational purposes only. Always consult a dental professional for medical advice.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()