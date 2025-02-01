---
title: "CIDAUT AI Fake Scene Classification 2024"
excerpt: "Classify if an autonomous driving scene is real or fake. [Website](https://www.kaggle.com/competitions/cidaut-ai-fake-scene-classification-2024/overview)<br/><img src='/images/inbox_2779868_f6a1889d486931be4fc03c08cd0c000e_37.jpg'>"
collection: portfolio
---

<img src='/images/inbox_2779868_f6a1889d486931be4fc03c08cd0c000e_37.jpg'>

Description:
------
Your task is to develop a neural network or apply a suitable algorithm to classify whether an image of a driving scenario is real or fake. The images are provided in RGB format and compressed as JPEG files. Each image is labeled with 1 for real and 0 for fake, indicating a binary classification problem. You are free to create your own train-validation split for model training and evaluation. However, for the test images, the labels are not available; refer to the sample_submission.csv file in the Data section for submission formatting. The code must be written in Python, and you can utilize frameworks such as TensorFlow, Keras, or PyTorch. Additionally, you are allowed to leverage public GitHub repositories, pre-trained models, and other publicly available datasets to enhance your solution.

Training model:
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torchmetrics.classification import BinaryF1Score

import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from transformers import AutoImageProcessor, AutoModelForImageClassification

# Focal Loss implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return torch.mean(F_loss)

# Custom dataset
class CustomImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = csv_file
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        label = torch.tensor(int(self.annotations.iloc[idx, 1]))

        if self.transform:
            image = self.transform(image)
            
        return image, label

# Data augmentation transformations
def get_transform(img_size=(512, 512)):
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
    ])

# Create data loaders
def create_dataloaders(csv_file, img_dir, img_size=(512, 512), batch_size=32, n_fold=0):
    transform = get_transform(img_size)
    dataset = CustomImageDataset(csv_file=csv_file, img_dir=img_dir, transform=transform)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for i, (train_index, val_index) in enumerate(skf.split(np.zeros(len(csv_file)), csv_file.iloc[:, 1].values)):
        if i == n_fold:
            break
            
    train_dataset = Subset(dataset, train_index)
    val_dataset = Subset(dataset, val_index)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    
    return train_loader, val_loader

# Single epoch training
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images).logits[:, :1]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(train_loader.dataset)

# Validation
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_outputs = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(images).logits[:, :1]
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            all_labels.append(labels.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())

    epoch_loss = running_loss / len(val_loader.dataset)
    all_labels = np.concatenate(all_labels)
    all_outputs = np.concatenate(all_outputs)
    all_outputs = torch.sigmoid(torch.tensor(all_outputs)).numpy()
    
    return epoch_loss, all_labels, all_outputs

# Early stopping
class EarlyStopping:
    def __init__(self, patience, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'Early stopping count: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), os.path.join(path, 'checkpoint1.pth'))
        self.val_loss_min = val_loss

# Main training function
def train_model(csv_file, img_dir, model, model_name, img_size=(512, 512), 
                num_epochs=100, batch_size=2, lr=1e-5, n_fold=0, 
                device='cuda', patience=10):
    train_loader, val_loader = create_dataloaders(csv_file, img_dir, 
                                                  img_size=img_size, 
                                                  batch_size=batch_size, 
                                                  n_fold=n_fold)

    model = model.to(device)
    
    # Compute class weights
    class_counts = csv_file['label'].value_counts()
    total_samples = len(csv_file)
    class_weights = total_samples / (2 * class_counts)
    pos_weight = torch.tensor(class_weights[1]).to(device)

    # Loss function
    criterion = FocalLoss()
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), 
                             lr=lr, 
                             weight_decay=1e-5)  # L2 regularization
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, 
                                   mode='min', 
                                   factor=0.5, 
                                   patience=3, 
                                   min_lr=1e-6,
                                   verbose=True)

    early_stopping = EarlyStopping(patience=patience, verbose=True)

    train_losses, val_losses = [], []
    
    path = f'{model_name}{n_fold}'
    os.makedirs(path, exist_ok=True)
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_labels, val_outputs = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Performance metrics
        val_preds = (val_outputs > 0.5).astype(int)
        accuracy = accuracy_score(val_labels, val_preds)
        f1 = f1_score(val_labels, val_preds)
        roc_auc = roc_auc_score(val_labels, val_outputs)
        
        print(f'Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
        print(f'Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}')
        
        # Learning rate adjustment
        scheduler.step(val_loss)
        
        # Early stopping
        early_stopping(val_loss, model, path)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='train loss')
    plt.plot(val_losses, label='val loss')
    plt.title('Model training loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(os.path.join(path, 'loss_plot.png'))
    plt.close()

    return model

# Main execution
if __name__ == '__main__':
    # Load pretrained model
    processor = AutoImageProcessor.from_pretrained("microsoft/beit-large-patch16-512")
    model = AutoModelForImageClassification.from_pretrained("microsoft/beit-large-patch16-512")

    # Load labels
    labels = pd.read_csv("train.csv")
    labels["label"] = labels["label"].map({"editada": 0, "real": 1})
    img_dir = "Train"

    # Training parameters
    batch_size = 2
    lr = 1e-5
    img_size = (512, 512)
    n_fold = 0

    # Start training
    trained_model = train_model(
        labels, 
        img_dir, 
        model, 
        'microsoft/beit-large-patch16-512', 
        img_size=img_size, 
        num_epochs=100, 
        batch_size=batch_size, 
        lr=lr, 
        n_fold=n_fold, 
        patience=10
    )

# Clear GPU memory
torch.cuda.empty_cache()
```

Make predictions:
```python
import torch
import pandas as pd
import os
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

def create_submission(model_path, test_dir, output_csv='submission.csv'):
    # Check if the device supports GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device in use: {device}")

    # Load the image processor and model
    processor = AutoImageProcessor.from_pretrained("microsoft/beit-large-patch16-512")
    model = AutoModelForImageClassification.from_pretrained("microsoft/beit-large-patch16-512")
    
    # Load the trained weights
    model.load_state_dict(torch.load(os.path.join(model_path, 'checkpoint1.pth')))
    model = model.to(device)  # Move model to GPU
    model.eval()  # Set model to evaluation mode

    # Prepare a list to store prediction results
    predictions = []

    # Iterate over images in the test directory
    for filename in sorted(os.listdir(test_dir)):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            # Full image path
            image_path = os.path.join(test_dir, filename)
            
            # Open and convert the image
            image = Image.open(image_path).convert("RGB")
            
            # Preprocess the image
            inputs = processor(image, return_tensors="pt").to(device)  # Transfer data to GPU
            
            # Make predictions
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.sigmoid(outputs.logits).cpu().numpy()[0][0]  # Transfer output back to CPU for further processing
            
            # Convert probabilities to class (1 for real, 0 for fake)
            predicted_class = 1 if probabilities > 0.5 else 0
            
            # Add to prediction results
            predictions.append({'image': filename, 'label': predicted_class})

    # Create a DataFrame
    submission_df = pd.DataFrame(predictions)
    
    # Save as CSV
    submission_df.to_csv(output_csv, index=False)
    
    print(f"Submission CSV saved to {output_csv}")
    print("Preview of prediction results:")
    print(submission_df)

# Example usage
model_path = 'microsoft/beit-large-patch16-5120'  # Folder containing model weights
test_dir = 'Test'  # Folder containing test images
create_submission(model_path, test_dir)
```