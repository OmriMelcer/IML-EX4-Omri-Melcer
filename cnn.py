import time
import os
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torch
import torchvision
from tqdm import tqdm
from torchvision import transforms
import numpy as np

class ResNet18(nn.Module):
    def __init__(self, pretrained=False, probing=False):
        super(ResNet18, self).__init__()
        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
            self.transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
            self.resnet18 = resnet18(weights=weights)
        else:
            self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])            
            self.resnet18 = resnet18()
                
        
        in_features_dim = self.resnet18.fc.in_features        
        self.resnet18.fc = nn.Identity()
        if probing:
            for name, param in self.resnet18.named_parameters():
                    param.requires_grad = False
        self.logistic_regression = nn.Linear(in_features_dim, 1)

    def forward(self, x):
        features = self.resnet18(x)
        return self.logistic_regression(features)

def get_loaders(path, transform, batch_size):
    """
    Get the data loaders for the train, validation and test sets.
    :param path: The path to the 'whichfaceisreal' directory.
    :param transform: The transform to apply to the images.
    :param batch_size: The batch size.
    :return: The train, validation and test data loaders.
    """
    train_set = torchvision.datasets.ImageFolder(root=os.path.join(path, 'train'), transform=transform)
    val_set = torchvision.datasets.ImageFolder(root=os.path.join(path, 'val'), transform=transform)
    test_set = torchvision.datasets.ImageFolder(root=os.path.join(path, 'test'), transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def compute_accuracy(model, data_loader: torch.utils.data.DataLoader, device):
    """
    Compute the accuracy of the model on the data in data_loader
    :param model: The model to evaluate.
    :param data_loader: The data loader.
    :param device: The device to run the evaluation on.
    :return: The accuracy of the model on the data in data_loader
    """
    model.eval()
    ### YOUR CODE HERE ###
    total_correct = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            predictions = model(batch_X) # Shape: [Batch_Size, 1]
            # Threshold at 0: logits > 0 is class 1 (Real), else class 0 (Fake)
            predicted_labels = (predictions.squeeze() > 0).float() 
            total_correct += (predicted_labels == batch_y).sum().item()
            total_samples += batch_y.size(0)

    accuracy = total_correct / total_samples if total_samples > 0 else 0
    return accuracy 

def run_training_epoch(model, criterion, optimizer, train_loader, device):
    """
    Run a single training epoch
    :param model: The model to train
    :param criterion: The loss function
    :param optimizer: The optimizer
    :param train_loader: The data loader
    :param device: The device to run the training on
    :return: The average loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    for (imgs, labels) in tqdm(train_loader, total=len(train_loader)):
        imgs = imgs.to(device)
        labels = labels.to (device)
        optimizer.zero_grad()
        predictions = model(imgs)
        loss = criterion(predictions, labels.float().unsqueeze(1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    avg_loss = total_loss / len(train_loader)
    return avg_loss

# Set the random seed for reproducibility
torch.manual_seed(0)

### UNCOMMENT THE FOLLOWING LINES TO TRAIN THE MODEL ###
# From Scratch
model = ResNet18(pretrained=False, probing=False)
# Linear probing
# model = ResNet18(pretrained=True, probing=True)
# Fine-tuning
# model = ResNet18(pretrained=True, probing=False)

transform = model.transform
batch_size = 32
num_of_epochs = 1
learning_rate = 0.0001
path = 'whichfaceisreal' # For example '/cs/usr/username/whichfaceisreal/'
train_loader, val_loader, test_loader = get_loaders(path, transform, batch_size)

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device =  torch.device("mps")
else:
    device =torch.device("cpu")
model = model.to(device)
### Define the loss function and the optimizer
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
### Train the model
start = time.time()
# Train the model
for epoch in range(num_of_epochs):
    # Run a training epoch
    loss = run_training_epoch(model, criterion, optimizer, train_loader, device)
    # Compute the accuracy
    train_acc = compute_accuracy(model, train_loader, device)
    # Compute the validation accuracy
    val_acc = compute_accuracy(model, val_loader, device)
    print(f'Epoch {epoch + 1}/{num_of_epochs}, Loss: {loss:.4f}, Val accuracy: {val_acc:.4f}')
    # Stopping condition
    ### YOUR CODE HERE ###
end = time.time()
# Compute the test accuracy
test_acc = compute_accuracy(model, test_loader, device)
print (test_acc)
print (end-start)

