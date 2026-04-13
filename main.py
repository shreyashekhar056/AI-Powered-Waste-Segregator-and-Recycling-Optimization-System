import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import os

class Dataset:
    def __init__(self, train_dir, val_dir, batch_size):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size

        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0)), 
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_loaders(self):
        train_dataset = datasets.ImageFolder(self.train_dir, transform=self.train_transform)
        val_dataset = datasets.ImageFolder(self.val_dir, transform=self.val_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        return train_loader, val_loader, train_dataset.classes

class Trainer:
    def __init__(self, model, device, lr, patience):
        self.model = model
        self.device = device
        self.patience = patience
        self.criterion = nn.CrossEntropyLoss()
        
        # IMPROVED: Lowered weight_decay from 1e-3 to 5e-4 to allow faster initial learning
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        
        self.scheduler = None
        self.best_accuracy = 0.0
        self.early_stopping_counter = 0

    def set_scheduler(self, train_loader, num_epochs):
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.optimizer.param_groups[0]['lr'],
            epochs=num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.2, # Start annealing earlier to stabilize learning
            anneal_strategy='cos'
        )

    def train_one_epoch(self, train_loader):
        self.model.train()
        total_loss, total_correct, total_samples = 0, 0, 0
        for data, targets in train_loader:
            data, targets = data.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()
                
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(targets).sum().item()
            total_samples += targets.size(0)
            
        return total_loss / len(train_loader), 100. * total_correct / total_samples

    def train(self, train_loader, val_loader, num_epochs, validator, saver, class_names):
        self.set_scheduler(train_loader, num_epochs)
        for epoch in range(num_epochs):
            print(f"\nEpoch [{epoch+1}/{num_epochs}]")
            train_loss, train_acc = self.train_one_epoch(train_loader)
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            
            val_loss, val_acc = validator.validate(val_loader, self.model, self.device, self.criterion)
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                saver.save(self.model, self.optimizer, epoch, self.best_accuracy, class_names)
                print(f"*** Best model updated: {self.best_accuracy:.2f}% ***")
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1

            if self.early_stopping_counter >= self.patience:
                print(f"Stopped early. Best Val Acc: {self.best_accuracy:.2f}%")
                break

class Validation:
    def validate(self, val_loader, model, device, criterion):
        model.eval()
        total_loss, total_correct, total_samples = 0, 0, 0
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total_correct += predicted.eq(targets).sum().item()
                total_samples += targets.size(0)
        return total_loss / len(val_loader), 100. * total_correct / total_samples

class ModelSaver:
    @staticmethod
    def save(model, optimizer, epoch, best_accuracy, class_names, save_path="saved_models/best_model.pth"):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_accuracy': best_accuracy,
            'class_names': class_names
        }, save_path)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    data_loader = Dataset("dataset/train", "dataset/val", batch_size=32)
    train_loader, val_loader, class_names = data_loader.get_loaders()
    num_classes = len(class_names)
    print(f"Detected Classes: {class_names}")

    print("Initializing ResNet18...")
    model = models.resnet18(weights='DEFAULT')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    # IMPROVED: Increased LR to 1e-4 and Patience to 12
    learning_rate = 1e-4  
    num_epoch = 50
    patience = 12 
    
    trainer = Trainer(model, device, lr=learning_rate, patience=patience)
    trainer.train(train_loader, val_loader, num_epoch, Validation(), ModelSaver(), class_names)

if __name__ == "__main__":
    main()