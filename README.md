# Teeth Classification Project

A deep learning project for the classification of dental images into seven distinct categories. This model is designed to support AI-driven dental solutions, aiming to enhance diagnostic accuracy and improve patient outcomes in the healthcare industry.

![Project Banner](https://placehold.co/1200x400/007BFF/FFFFFF?text=Teeth+Classification+AI)

## 📖 Table of Contents
- [Project Objective](#-project-objective)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#-usage)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Training](#model-training)
  - [Evaluation](#evaluation)
- [Model Architecture](#-model-architecture)
- [Results](#-results)
- [Future Work](#-future-work)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

## 🎯 Project Objective

The primary goal of this project is to develop a robust computer vision model capable of accurately classifying dental images into 7 distinct categories: **CaS, CoS, Gum, MC, OC, OLP, and OT**.

The key objectives include:
-   **Preprocessing:** Preparing dental images for analysis through normalization and augmentation to ensure optimal quality for model training.
-   **Visualization:** Analyzing the class distribution and the effects of data augmentation.
-   **Model Development:** Building and training a custom deep learning model from scratch using TensorFlow to establish a strong performance baseline.

## 📁 Dataset

The dataset is sourced from a collection of dental images and is organized into three main directories for training, validation, and testing.

-   **`Training/`**: Contains images used for training the model.
-   **`Validation/`**: Contains images used for validating the model's performance during training.
-   **`Testing/`**: Contains images used for the final evaluation of the trained model.

Each directory contains subfolders for the 7 classes, ensuring a clear and organized structure.

## 🏗️ Project Structure

teeth-classification-project/
│
├── data/
│   ├── Training/
│   ├── Validation/
│   └── Testing/
│
├── notebooks/
│   └── Teeth_Classification_Project_V0_2.ipynb
│
├── saved_models/
│   └── my_model.h5
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt


## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

Ensure you have Python 3.8+ installed. You will also need `pip` for installing the required packages.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/teeth-classification-project.git](https://github.com/your-username/teeth-classification-project.git)
    cd teeth-classification-project
    ```
2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## ⚙️ Usage

The core logic for data processing, model training, and evaluation is contained within the Jupyter Notebook.

### Data Preprocessing
The notebook `Teeth_Classification_Project_V0_2.ipynb` contains detailed steps for loading the dataset, resizing images to a uniform size, converting them to NumPy arrays, and normalizing pixel values to a range of [0, 1].

### Model Training
Run the cells in the notebook sequentially to train the model on the preprocessed data. The training process includes data augmentation to improve generalization.

### Evaluation
After training, the model is evaluated on the test set. The notebook displays key performance metrics, including accuracy, a confusion matrix, and a detailed classification report.

## 🧠 Model Architecture

The model is a Convolutional Neural Network (CNN) built from scratch using the TensorFlow/Keras library. The architecture is specifically tailored for this image classification task and consists of several convolutional and pooling layers, followed by dense layers for classification. This custom architecture serves as a performance baseline for future iterations.

## 📊 Results

The model achieves a high level of performance on the test dataset.

-   **Accuracy:** **96%**

The detailed classification report in the notebook provides a breakdown of precision, recall, and F1-score for each of the 7 classes, demonstrating the model's effectiveness in distinguishing between different dental conditions.

## 💡 Future Work

Potential areas for future development include:
-   Experimenting with more complex architectures like ResNet or EfficientNet.
-   Implementing advanced data augmentation techniques.
-   Deploying the trained model as a web application or API for real-time inference.
-   Expanding the dataset to include more classes or a larger number of images.

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

Please see the `CONTRIBUTING.md` file for details on our code of conduct and the process for submitting pull requests.

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## 📧 Contact

Your Name - [your.email@example.com](mailto:your.email@example.com)

Project Link: [https://github.com/your-username/teeth-classification-project](https://github.com/your-username/teeth-classification-project)
# Teeth-Classification-Project
# Teeth-Classification-Project
