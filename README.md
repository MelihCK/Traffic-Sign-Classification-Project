# 🚦 Traffic Sign Classification Project

## 📜 Overview

This project is conducted as part of the TKPR221 Fundamentals of Machine Learning course at METU. The aim is to classify traffic signs using various machine learning techniques, including classical models, convolutional neural networks (CNN), ensemble learning, and transfer learning. The Mapillary Traffic Sign Dataset is used for training and evaluation.

## 👨‍💻 Contributors

We've worked on this project as a group: <br>
<a href="https://github.com/muguryalcin">Mustafa Uğur Yalçın</a>,<br> 
<a href="https://github.com/MelihCK">Melih Can Kanmaz</a>, <br>
<a href="https://github.com/topraktoprak1">Refik Toprak Telli</a><br>

## 🗂️ Dataset

The dataset used is The Mapillary Traffic Sign Dataset for Detection and Classification on a Global Scale. The first fully annotated part, which includes approximately 12,000 images, was utilized. After cropping bounding boxes, a total of 59,552 traffic sign images were obtained.

## ⚙️ Methodology

### 🧹 Data Preprocessing

Cropping traffic signs using bounding boxes.

Resizing and padding images for uniform input sizes.

Data augmentation techniques applied:

- RandomBrightnessContrast

- HorizontalFlip

- VerticalFlip

- Normalize

- Rotate

Handling class imbalance using SMOTE.

### 🔹 Model Selection

Classical Machine Learning Models

Used SGDClassifier with hyperparameter tuning.

Performed relatively poorly due to limitations in handling image data.

### CNN + Machine Learning

Extracted features using CNN, then used classical ML classifiers.

### CNN Models

Built a custom CNN architecture with multiple layers.

Applied Adam optimizer with a learning rate of 0.001.

### Ensemble Learning

Combined predictions from multiple models.

Utilized Random Forest, Decision Tree, and XGBoost.

### Transfer Learning

Used ResNet50 pre-trained model and fine-tuned it on the dataset.

Achieved the best results among all approaches.

## 🔹 Experimentation

Used Google Colab for model training.

Tracked experiment metrics using Neptune.ai.

Applied hyperparameter tuning and cross-validation to optimize models.

## 📊 Results

| Model                          | Best Accuracy | Best F1 Score |
|---------------------------------|---------------|---------------|
| Classical ML (SGD)             | ~0.12         | ~0.1          |
| CNN + ML                        | ~0.75         | ~0.76         |
| CNN                             | ~0.97         | ~0.97         |
| Transfer Learning (ResNet50)    | ~0.96         | ~0.96         |
| Ensemble Learning               | ~0.725        | ~0.2          |


## 🔑 Key Findings

- Transfer learning (ResNet50) achieved the highest and most consistent performance.

- Data augmentation and SMOTE significantly improved model accuracy.

- CNN-based approaches outperformed classical ML models in image classification.

## 🛠️ Installation & Usage

### 🖥️ Clone the repository
'git clone https://github.com/yourusername/Traffic-Sign-Classification.git'
'cd Traffic-Sign-Classification'

### ⚙️ Install dependencies
'pip install -r requirements.txt'

### ▶️ Run the model training script
'python train_model.py'

## 🔮 Future Work

- Utilize larger datasets to improve model generalization.

- Apply other deep learning models such as Vision Transformers.

- Enhance ensemble learning approaches for better accuracy.

## 📜 License

This project is open-source and distributed under the MIT License.
