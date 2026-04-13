import torch
import torch.nn as nn
from torchvision import models
import cv2
import numpy as np
import time
import os

class WebcamPredictor:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Check your path! Can't find: {model_path}")

        # 1. Load the checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.class_names = checkpoint['class_names']
        num_classes = len(self.class_names)

        # 2. Initialize ResNet18 (Must match the architecture in main.py)
        print("Initializing ResNet18 Architecture...")
        self.model = models.resnet18()
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

        # 3. Load the weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded successfully with {num_classes} classes.")

    def preprocess_frame(self, frame):
        # Match the normalization used in training
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img).unsqueeze(0).to(self.device)

    def start_webcam(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not access webcam.")
            return

        # Set resolution for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("Webcam Live! Press 'q' to quit.")
        prev_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret: break

            input_tensor = self.preprocess_frame(frame)
            
            with torch.no_grad():
                output = self.model(input_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)
                confidence, idx = torch.max(probs, dim=1)
            
            label = self.class_names[idx.item()]
            conf_percent = confidence.item() * 100

            # UI Overlay
            # Green for high confidence (>70%), Yellow otherwise
            color = (0, 255, 0) if conf_percent > 70 else (0, 255, 255)
            
            # Draw a box/label background
            cv2.rectangle(frame, (0, 0), (350, 80), (0, 0, 0), -1)
            cv2.putText(frame, f"CLASS: {label.upper()}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"CONF: {conf_percent:.1f}%", (10, 65), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # FPS Calculation
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 450), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow('ResNet18 Waste Classifier', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    MODEL_FILE = "saved_models/best_model.pth"
    try:
        predictor = WebcamPredictor(MODEL_FILE)
        predictor.start_webcam()
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")