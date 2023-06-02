import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split

# Set the random seed for reproducibility
torch.manual_seed(42)

# Define the dataset and data loaders
dataset_path = "../../pics/2750/"
transform = ToTensor()

# Load the main dataset
dataset = ImageFolder(dataset_path, transform=transform)

# Create a subset containing specific classes (indices)
class_indices = [1, 4]  # Replace with the desired class indices
forest_industrial_dataset = torch.utils.data.Subset(dataset, class_indices)

# Split the subset into train and validation sets
train_ratio = 0.8
train_size = int(train_ratio * len(forest_industrial_dataset))
val_size = len(forest_industrial_dataset) - train_size

train_dataset, val_dataset = random_split(forest_industrial_dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 13 * 13, 120)
        self.drop1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(120, 84)
        self.drop2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(84, 2)
        
    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        return self.fc3(x)  

# Initialize the model
model = CNN()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Set the number of epochs
num_epochs = 20

# Training loop
epoch_losses = []
for epoch in range(num_epochs):
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
    epoch_losses.append(epoch_loss)
    print(f'Epoch: {epoch+1}/{num_epochs} | Loss: {epoch_loss:.4f}')

# Plot the loss curve
plt.plot(epoch_losses, 'r')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()
