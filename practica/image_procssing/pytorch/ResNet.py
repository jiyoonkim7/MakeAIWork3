import torch
import torch.optim as optim
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.io import ImageReadMode, read_image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob

forestDirectory = '../../pics/2750/River'
forest_images = glob.glob(forestDirectory + '/*.jpg')

industrialDirectory = '../../pics/2750/Industrial'
industrial_images = glob.glob(industrialDirectory + '/*.jpg')

#print(forest_images)
#print(industrial_images)

industrial_labels = [0] * len(industrial_images)  # label 0 for industrial images
forest_labels = [1] * len(forest_images)  # label 1 for forest images

X = industrial_images + forest_images
Y = industrial_labels + forest_labels

class ImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_path = self.images[index]
        label = self.labels[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Define the transformations
transform = T.Compose([
    T.ToTensor(),
    T.Resize((32, 32)),  # Resize the images to a consistent size
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the image tensors
])

# Create the dataset
dataset = ImageDataset(X, Y, transform=transform)

# Define the train-validation split ratio (e.g., 0.8 for 80% training, 0.2 for 20% validation)
train_ratio = 0.8
train_size = int(train_ratio * len(dataset))
val_size = len(dataset) - train_size

# Split the dataset into train and validation sets
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create dataloaders for train and validation sets
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize the ResNet model
model = torchvision.models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

# Adjust the last linear layer of the ResNet model
num_classes = 2  # Number of classes (2 in this case: industrial and forest)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Set the number of epochs
num_epochs = 100

# Training loop
for epoch in range(num_epochs):
    # Training
    model.train()
    running_loss = 0.0
    for images, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(train_dataloader)
    
    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")
