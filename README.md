# Teeth Classification Project

A deep learning project for the classification of dental images into seven distinct categories. This model supports AI-driven dental diagnostics and aims to enhance diagnostic accuracy and improve patient outcomes in the healthcare industry.

![Project Banner](https://placehold.co/1200x400/007BFF/FFFFFF?text=Teeth+Classification+AI)

---

## Table of Contents
- [Project Objective](#project-objective)
- [Model Overview](#model-overview)
- [Model Comparison](#model-comparison)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Training](#model-training)
  - [Evaluation](#evaluation)
- [Web Deployment](#web-deployment)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Project Objective

The goal of this project is to develop a robust computer vision model capable of classifying dental images into seven categories:

- CaS (Caries Susceptibility)
- CoS (Composite Restoration)
- Gum (Gingival Tissue)
- MC (Metal Crown)
- OC (Oral Cancer)
- OLP (Oral Lichen Planus)
- OT (Other)

---

## Model Overview

Two deep learning models were developed and compared:

- Custom CNN (built from scratch using TensorFlow/Keras)
- Transfer Learning model using MobileNetV2

---

## Model Comparison

| Model             | Architecture        | Test Accuracy | Key Advantage                         |
|------------------|---------------------|---------------|---------------------------------------|
| Custom CNN        | Built from Scratch  | 96%           | Highest accuracy on test data         |
| Transfer Learning | MobileNetV2         | 94%           | Faster inference, smaller size        |

Deployment Decision: The Custom CNN model was selected for deployment.

---

## Dataset

The dataset is organized into:

- `Training/` - Images used to train the model
- `Validation/` - Used for hyperparameter tuning and early stopping
- `Testing/` - Used to evaluate model performance after training

Each folder contains subfolders for each of the seven classes.

---

## Project Structure

```text
teeth-classification-project/
│
├── data/
│   └── Teeth_Dataset/
│       ├── Training/
│       ├── Validation/
│       └── Testing/
│
├── notebooks/
│   ├── Teeth_Classification_Project.ipynb      # Custom CNN
│   └── Teeth_TransferLearning.ipynb            # MobileNetV2
│
├── saved_models/
│   ├── best_teeth_model.h5                     # Custom CNN model
│   └── my_model.h5                             # Alternative version
│
├── Teeth-Classification-Streamlit/
│   ├── app.py                                  # Streamlit app
│   ├── app_1.py                                # Experimental version
│   ├── images/                                 # Assets
│   ├── config.toml
│   ├── requirements.txt
│   └── runtime.txt
│
├── report/
│   └── Teeth_Classification.pdf
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```
## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip installed

### Installation

```bash
git clone https://github.com/your-username/teeth-classification-project.git
cd teeth-classification-project

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```bash
pip install -r requirements.txt
```
## Usage

### Data Preprocessing

The preprocessing pipeline is located in `Teeth_Classification_Project.ipynb`.

It includes:
- Image resizing and normalization
- Conversion to NumPy arrays
- Data augmentation to improve generalization

### Model Training

To train the model:

- Run the notebook cells sequentially
- You can choose between:
  - A Custom CNN built from scratch
  - A Transfer Learning model using MobileNetV2

Validation accuracy is used for:
- Early stopping
- Model checkpointing

### Evaluation

After training, the model is evaluated on the test set using:
- Test accuracy
- Confusion matrix
- Classification report (precision, recall, F1-score)

---

## Web Deployment

The best model is deployed using **Streamlit**.

**Features:**
- Upload dental images
- Get real-time predictions
- Clean and responsive UI

**To run locally:**

```bash
cd Teeth-Classification-Streamlit
streamlit run app.py
```
You can optionally deploy the app using platforms like:
- Streamlit Cloud
- Render
- Heroku

---

## Results

- **Best Test Accuracy:** 96% (Custom CNN)

The notebook includes:
- Class-wise precision, recall, and F1-score
- Visualizations for training and validation performance

---

## Future Work

- Experiment with deeper architectures (e.g., ResNet, EfficientNet)
- Apply advanced data augmentation techniques
- Deploy the model as a cloud API or mobile app
- Increase dataset diversity and size

---

## Contributing

Contributions are welcome and appreciated.

**How to contribute:**
1. Fork the repository
2. Create a new feature branch
3. Commit your changes
4. Open a pull request on GitHub

---

## License

This project is licensed under the MIT License.  
See the `LICENSE` file for full details.

---

## Contact

**Hassan Abdul-razeq**  
Email: [ha.razak.ds@gmail.com](mailto:ha.razak.ds@gmail.com)  
LinkedIn: [https://linkedin.com/in/hassan-abdul-razeq](https://linkedin.com/in/hassan-abdul-razeq)

