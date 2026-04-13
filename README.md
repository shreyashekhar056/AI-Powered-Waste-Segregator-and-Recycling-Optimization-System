# AI-Powered Waste Segregator & Recycling System 
An automated waste classification system built with Deep Learning (PyTorch) and ResNet-18. This project is designed to accurately categorize waste into 6 distinct classes, enabling smarter recycling and waste management.

**The model predicts the type of waste from an image and can be used for smart recycling or educational applications.**


---

## Highlights

** Optimized Dataset: Focused on 6 high-impact waste categories, ensuring high precision and practical applicability.

**High Performance: Achieved a peak 100% Validation Accuracy using transfer learning and customized training schedulers.

**Integrated Visualization: Built-in plot_metrics functionality within the training pipeline to generate accuracy and loss graphs automatically.

**Transfer Learning Power: Leverages the ResNet-18 architecture, fine-tuned specifically for waste material recognition.

**Git LFS Integration: Large model weights (.pth files) are managed via Git Large File Storage for seamless repository management.

---

## Table of Contents

## Table of Contents

* [Project Structure](#project-structure)
* [Setup & Installation](#setup--installation)
* [Dataset](#dataset)
* [Model Architecture](#model-architecture)
* [Training](#training)
* [Usage Example](#usage-example)
* [Results](#results)

---

## Project Structure

```
## Project Structure

```text
AI-Powered-Waste-Segregator/
├── dataset/             # Organized into train/val splits
│   ├── train/           # Cardboard, Glass, Metal, Paper, Plastic, Trash
│   └── val/             # Validation images per category
├── saved_models/        # Storage for the best-performing model weights
│   └── best_model.pth   (Managed via Git LFS)
├── Waste_Training.ipynb # Primary notebook with auto-graphing logic
├── app.py               # Streamlit web interface for deployment
├── main.py               # Core training script
├── predict_image.py     # Inference script for single image testing
└── requirements.txt     # List of dependencies (PyTorch, Torchvision, etc.)
```

* `main.py`: Trains the CNN model for 50 epochs and saves the best model to `saved_models/best_model.pth`.
* `object-detection.py`:carries out inference for garbage classification based on webcams or images and forecasts manual inspection.
* `validation-checker.py`: Evaluates the model, computes 89.56% accuracy, and generates a confusion matrix.
* `validation-splitter.py`: Splits the dataset into 80% training and 20% validation sets.
* `requirements.txt`: Lists dependencies like PyTorch, OpenCV, and NumPy.

---

## Setup & Installation

Follow these steps to set up and run the project:

```bash
# Clone the repository
git clone https://github.com/shreyashekhar056/AI-Powered-Waste-Segregator-and-Recycling-Optimization-System.git
cd AI-Powered-Waste-Segregator-and-Recycling-Optimization-System

# Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Training Notebook or Script
python main.py

# Launch the Prediction Web App
streamlit run app.py
```

---

## Dataset

The dataset is a modified version of the Kaggle Garbage Classification dataset, tailored to 10 waste categories:

Source : [Kaggle dataset](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)

* Battery
* Cardboard
* Glass
* Metal
* Paper
* Plastic
* Trash

The dataset is split into:

* **Training Set:** 80% of the data.
* **Validation Set:** 20% of the data.

Images are resized to 224x224 pixels to match the CNN’s input requirements.

---

## Model Architecture

The CNN is designed for efficiency and accuracy, with six convolutional layers followed by fully connected layers. Below is the model definition:

```python
import torch
import torch.nn as nn
from torchvision import models

# Load the pretrained ResNet18 backbone
model = models.resnet18(weights='DEFAULT')

# Fine-tuning: Replace the final fully connected layer
# num_ftrs matches the 512 output neurons of the ResNet backbone
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes) # num_classes = 6
```

**Summary Table:**

<img width="903" height="389" alt="image" src="https://github.com/user-attachments/assets/b81fb713-4bd3-4147-bbe7-cdb059661bd4" />


**Neuron Count**

* **Convolutional Layers:** Feature maps increase progressively (16, 32, 64, 128, 256, 512).
* **Fully Connected Layers:**

  * First FC: 512 × 3 × 3 = 4608 inputs → 512 outputs.
  * Output FC: 512 inputs → 10 outputs (num\_classes).

**Input Processing**

* Input: 224x224 RGB images.
* After six MaxPool2d(2,2) layers: 224 → 112 → 56 → 28 → 14 → 7 → 3 (rounded down).

---

## Training

The model was trained with the following hyperparameters:

* Number of Classes: 10
* Batch Size: 8
* Learning Rate: 5e-4
* Epochs: 50
* Early Stopping Patience: 10
* Optimizer: AdamW
* Loss Function: Cross Entropy Loss

Run `main.py` to train the model for 50 epochs. The best model is saved to `saved_models/best_model.pth`.

---

## Evaluation

The model was evaluated using `validation-checker.py`, which:

* Achieves a validation accuracy of **89.56%**.
* Generates a confusion matrix to analyze classification performance across categories.

Predictions are supported for both webcam feeds and static images via `object-detection.py`.


---

## Usage Example

**Train the Model:**

```bash
python3 main.py
```

**Predict with Webcam or Image:**

```bash
python3 object-detection.py
```

**Example:**
Below is the updated `predict_frame` function from `object-detection.py`, addressing the type mismatch error (Input type (double) and bias type (float) should be the same):

```python
def predict_frame(self, frame):
    try:
        import cv2
        import numpy as np
        import torch
        
        # 1. Preprocessing: Convert BGR to RGB and resize to 224x224
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        
        # 2. Normalization: Apply ImageNet mean and std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = img.astype(np.float32) / 255.0
        img = (img - mean) / std
        
        # 3. Tensor Conversion: HWC to CHW and add batch dimension
        img = np.transpose(img, (2, 0, 1))
        img_tensor = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        
        # 4. Inference
        with torch.no_grad():
            output = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_idx].item()
            predicted_class = self.class_names[predicted_idx]
        
        return predicted_class, confidence
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None, None
```

**Fix Explanation:** The error was resolved by ensuring the input tensor and model parameters use float32 precision. The mean and std arrays are set to np.float32, and `.float()` is explicitly applied to `img_tensor`.

**Example output:**

```
Predicted Class: Plastic
Confidence: 0.92
```

**Evaluate the Model:**

```bash
python3 validation-checker.py
```

Outputs:

* Validation accuracy: 89.5%
* Confusion matrix


---

## Results

The model achieved a validation accuracy of **89.5%**. Below are the detailed precision, recall, F1-score, and support for each class as calculated on the validation set:

**Visualizations:**

* Confusion Matrix: Shows classification performance across the 10 classes.
  
<img width="792" height="706" alt="image" src="https://github.com/user-attachments/assets/ff062656-b8a3-47a2-a59d-9f7e792d3762" />
  
* Webcam Prediction: Real-time classification from webcam feed.
  
 <img width="1903" height="890" alt="Screenshot 2026-04-13 004324" src="https://github.com/user-attachments/assets/692928fe-d5cf-471d-94f1-822e8d636c64" />

* Image Prediction: Accurate classification of static images.
  
 <img width="1911" height="857" alt="Screenshot 2026-04-13 004359" src="https://github.com/user-attachments/assets/f5f5e0a1-d380-4fce-ae3d-6ac9e92b8223" />



---

## References

* IBM Deep Learning with PyTorch Course
* Gyawali, D., Regmi, A., Shakya, A., Gautam, A., and Shrestha, S., 2020. Comparative analysis of multiple deep CNN models for waste classification. arXiv preprint arXiv:2004.02168.
* Kaggle Garbage Classification Dataset
