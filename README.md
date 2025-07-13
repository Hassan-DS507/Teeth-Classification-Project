
# Teeth Classification AI 

![Project Banner](https://placehold.co/1200x400/007BFF/FFFFFF?text=AI-Powered+Dental+Diagnostics)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://teeth-classification-project-hassan.streamlit.app)

An advanced deep learning system for classifying dental X-ray images into 7 clinical categories, providing real-time diagnostic suggestions with clinical explanations and recommendations.

##  Overview

This AI-powered diagnostic tool helps dental professionals and students quickly analyze dental images by:

- Classifying images into 7 clinical categories
- Providing confidence scores for predictions
- Offering detailed clinical descriptions
- Suggesting evidence-based recommendations
- Linking to trusted external resources

**Live Demo:** [https://teeth-classification-project-hassan.streamlit.app](https://teeth-classification-project-hassan.streamlit.app)

##  Features

- **Real-time AI Analysis**: Get instant classification results
- **Clinical Insights**: Detailed descriptions for each condition
- **Recommendations**: Actionable clinical suggestions
- **Trusted Resources**: Direct links to MouthHealthy.org
- **Visual UI**: Clean, professional interface with animations
- **Responsive Design**: Works on desktop and mobile

##  Tech Stack

| Component          | Technology |
|--------------------|------------|
| Frontend           | Streamlit |
| Deep Learning      | TensorFlow/Keras |
| Image Processing   | PIL, OpenCV |
| Data Handling      | NumPy, Pandas |
| Visual Enhancements| streamlit-lottie |
| Deployment         | Streamlit Cloud |

##  Classification Categories

The model identifies 7 dental conditions:

| Class | Icon | Condition Name | Description |
|-------|------|----------------|-------------|
| CaS  | Caries Superficial | Early stage tooth decay |
| CoS  | Composite Superficial | Shallow dental filling |
| Gum  | Gum Area | Gingival tissue focus |
| MC   | Metal Crown | Artificial metal restoration |
| OC  | Orthodontic Component | Braces/alignment devices |
| OLP  | Oral Lichen Planus | Chronic inflammatory condition |
| OT  | Other | Undefined condition |

##  Model Performance

### Custom CNN Model
- **Validation Accuracy**: 93%
- **Input Shape**: (224, 224, 3)
- **Architecture**:
  - Convolutional layers with ReLU activation
  - MaxPooling layers
  - Flatten + Dense layers
  - Softmax output

### MobileNetV2 Transfer Learning
- **Validation Accuracy**: 98%
- **Faster inference** with comparable performance

**Deployed Model**: Custom MobileNetV2 (`best_teeth_model_try2.h5`)

##  Demo Screenshots

| Upload Interface | Prediction Result | Clinical Recommendations |
|------------------|-------------------|--------------------------|
| ![Upload](https://placehold.co/600x400/003366/FFFFFF?text=Upload+Interface) | ![Result](https://placehold.co/600x400/006633/FFFFFF?text=Prediction+Result) | ![Recommendations](https://placehold.co/600x400/660033/FFFFFF?text=Clinical+Info) |

##  Project Structure

```text
Teeth-Classification-AI/
├── Teeth-Classification-Streamlit/
│   ├── app.py                  # Main Streamlit application
│   ├── final_model/            # Trained models
│   │   └── best_teeth_model_try2.h5
│   ├── requirements.txt        # Python dependencies
│   └── assets/                 # Static files
├── notebooks/                  # Jupyter notebooks
│   ├── Teeth_Classification_Project.ipynb
│   └── Teeth_TransferLearning.ipynb
├── data/                       # Dataset (not included in repo)
├── README.md                   # This file
└── LICENSE                     # MIT License
```

##  Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Hassan-DS507/Teeth-Classification-Project.git
   cd Teeth-Classification-AI/Teeth-Classification-Streamlit
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

##  External Resources

Each diagnosis includes links to trusted dental health resources:

- [Tooth Decay](https://www.mouthhealthy.org/all-topics-a-z/tooth-decay)
- [Dental Fillings](https://www.mouthhealthy.org/all-topics-a-z/fillings)
- [Gum Disease](https://www.mouthhealthy.org/all-topics-a-z/gum-disease)
- [Dental Crowns](https://www.mouthhealthy.org/all-topics-a-z/crowns)
- [Orthodontics](https://www.mouthhealthy.org/all-topics-a-z/orthodontics)
- [Oral Cancer](https://www.mouthhealthy.org/all-topics-a-z/oral-cancer)
- [Dental Visits](https://www.mouthhealthy.org/all-topics-a-z/dental-visits)

##  Disclaimer

This application is for **educational purposes only** and not intended as a substitute for professional dental diagnosis or treatment. Always consult with a qualified dental professional for medical advice.

##  Future Roadmap

- [ ] Add Arabic language support
- [ ] Implement multi-class prediction visualization
- [ ] Create video walkthrough/tutorial
- [ ] Expand dataset with more diverse cases
- [ ] Develop mobile app version

##  Author

**Hassan Abdel-Razeq**  
- Email: [ha.razak.ds@gmail.com](mailto:ha.razak.ds@gmail.com)  
- GitHub: [https://github.com/your-username](https://github.com/your-username)  
- LinkedIn: [https://linkedin.com/in/hassan-abdul-razeq](https://linkedin.com/in/hassan-abdul-razeq)  

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

