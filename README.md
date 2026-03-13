```markdown
# Brain Tumor Classification using MRI Images

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-2.x-red)
![License](https://img.shields.io/badge/License-MIT-green)

This project implements a deep learning model to classify brain MRI scans into four categories: **glioma**, **meningioma**, **pituitary tumor**, or **no tumor**. Using transfer learning with VGG16 and fine‑tuning with class weights, the model achieves **98% accuracy** on a test set of 2414 images.

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Model Architecture](#model-architecture)
- [Training Details](#training-details)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Streamlit Web App](#streamlit-web-app)
- [Future Improvements](#future-improvements)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Overview
Brain tumors are a serious medical condition requiring accurate and timely diagnosis. MRI is the primary imaging modality, but manual interpretation is time‑consuming and subject to inter‑observer variability. This project aims to assist radiologists by providing an automated classification system using deep learning. The model is trained on a public Kaggle dataset and can be used via a simple Streamlit interface.

---

## Dataset
We use the **Brain Tumor MRI Classification** dataset from Kaggle, which contains 7023 human brain MRI images divided into four classes:

| Class        | Training Images | Testing Images |
|--------------|-----------------|----------------|
| Glioma       | 3018            | 755            |
| Meningioma   | 2183            | 546            |
| No Tumor     | 1945            | 487            |
| Pituitary    | 2504            | 626            |

The dataset is already split into `Training` and `Testing` folders. Class imbalance (especially the minority class "No Tumor") is addressed during training using class weights.

[Link to dataset](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)

---

## Methodology
1. **Data Preprocessing**  
   - Images resized to 224×224 pixels (VGG16 input size).  
   - Pixel values normalized to [0,1] and then passed through VGG16's `preprocess_input` (zero‑centering per channel).  
   - Data augmentation (rotation, shift, shear, zoom, flip, brightness) applied only to training set to improve generalization.

2. **Transfer Learning with VGG16**  
   - Pre‑trained VGG16 (ImageNet weights) used as feature extractor.  
   - Custom classifier head added:  
     - GlobalAveragePooling2D  
     - Dense(256, ReLU)  
     - Dropout(0.5)  
     - Dense(4, Softmax)

3. **Handling Class Imbalance**  
   - `compute_class_weight('balanced')` from scikit‑learn calculates weights inversely proportional to class frequencies.  
   - Weights passed to `model.fit()` so the model pays more attention to minority classes.

4. **Two‑Phase Training**  
   - **Phase 1 (Feature Extraction)**: Only the new classifier layers are trained (VGG16 frozen).  
   - **Phase 2 (Fine‑tuning)**: Last 30 layers of VGG16 unfrozen and trained with a very low learning rate (1e-5).  
   - Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau.

---

## Model Architecture
```
Input (224,224,3)
│
▼
VGG16 (frozen initially, later fine‑tuned)
│
▼
GlobalAveragePooling2D
│
▼
Dense(256, activation='relu')
│
▼
Dropout(0.5)
│
▼
Dense(4, activation='softmax')
```

**Total params:** 16,812,868 (trainable after fine‑tuning: ~7.1M)

---

## Training Details
- **Optimizer**: Adam (lr=1e-4 for phase 1, lr=1e-5 for phase 2)
- **Loss**: Categorical crossentropy
- **Batch size**: 32
- **Epochs**: Phase 1 up to 20 (early stopping), Phase 2 up to 20
- **Hardware**: Trained on Kaggle's Tesla P100 GPU (16 GB VRAM)

---

## Results
On the held‑out test set (2414 images):

| Class        | Precision | Recall | F1‑Score | Support |
|--------------|-----------|--------|----------|---------|
| glioma       | 0.97      | 0.96   | 0.97     | 755     |
| meningioma   | 0.98      | 0.95   | 0.96     | 546     |
| notumor      | 0.98      | 1.00   | 0.99     | 487     |
| pituitary    | 0.98      | 1.00   | 0.99     | 626     |

**Overall accuracy**: 0.98  
**Macro avg**: 0.98 (precision/recall/f1)  
**Weighted avg**: 0.98 (precision/recall/f1)

**AUC‑ROC (one‑vs‑rest)**:
- glioma: 0.9942
- meningioma: 0.9979
- notumor: 0.9988
- pituitary: 1.0000

Confusion matrix:
![Confusion Matrix](images/confusion_matrix.png) *(you can add an actual image)*

---

## Installation
Clone this repository and install the required packages:

```bash
git clone https://github.com/Rahul-Rathodseven/brain-tumor-classification.git
cd brain-tumor-classification
pip install -r requirements.txt
```

**Requirements**:
- tensorflow >= 2.6
- numpy
- matplotlib
- seaborn
- scikit-learn
- pandas
- streamlit
- pillow

---

## Usage

### 1. Run the Jupyter Notebook
The notebook `Brain_Tumor_Classification.ipynb` contains the complete pipeline: data download, preprocessing, training, evaluation, and model saving.  
To run it in Kaggle or Google Colab, ensure you have uploaded your `kaggle.json` API key.

### 2. Predict on a Single Image
Use the provided function in `predict.py`:

```python
from predict import predict_image
from tensorflow.keras.models import load_model

model = load_model('brain_tumor_vgg16.keras')
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
img_path = 'path/to/image.jpg'
pred_class, confidence, probabilities = predict_image(img_path, model, class_names)
print(f'Predicted: {pred_class} with confidence {confidence:.2f}')
```

### 3. Evaluate on Test Set
The notebook already evaluates the model; you can also run evaluation using the saved model and test generator.

---

## Streamlit Web App
A simple interactive web app is included (`app.py`). To launch it:

```bash
streamlit run app.py
```

The app allows you to upload an MRI image and displays:
- Predicted class with confidence score
- Bar chart of probabilities for all four classes
- A clinical disclaimer

![Streamlit App Screenshot](images/app_screenshot.png) *(optional)*

---

## Future Improvements
- **Grad‑CAM**: Add heatmaps to highlight regions influencing the decision.
- **Test other architectures**: EfficientNet, ResNet, or Vision Transformers.
- **Ensemble methods**: Combine multiple models for increased robustness.
- **Deploy as REST API** using Flask or FastAPI for integration into hospital systems.
- **Collect more data** for the "No Tumor" class or use synthetic data (GANs).

---

## Acknowledgements
- Dataset: [Brain Tumor MRI Classification](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) by Sartaj Bhuvaji
- Pre‑trained VGG16: [Keras Applications](https://keras.io/api/applications/vgg/)
- Inspired by numerous medical imaging deep learning projects.

---

## License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
# brain_tumor_detection-copy
