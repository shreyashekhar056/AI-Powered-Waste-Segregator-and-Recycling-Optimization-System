import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

def predict_image(image_path, model_path):
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Load Checkpoint to get class names and info
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint['class_names']
    num_classes = len(class_names)

    # 3. Initialize the ResNet18 Model (Must match main.py)
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # 4. Load the trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # 5. Image Transformation (Must match validation transforms)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 6. Load and Predict
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted].item()
        
    print(f"Result: {class_names[predicted.item()]} ({confidence*100:.2f}%)")

if __name__ == "__main__":
    # Update 'test.jpg' to whatever image you want to test
    target_image = "test.jpg" 
    if os.path.exists(target_image):
        predict_image(target_image, "saved_models/best_model.pth")
    else:
        print(f"Error: File '{target_image}' not found.")