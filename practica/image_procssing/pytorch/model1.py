# Define the train-validation split ratio (e.g., 0.8 for 80% training, 0.2 for 20% validation)
train_ratio = 0.8
train_size = int(train_ratio * len(subset))
val_size = len(subset) - train_size

# Verify that the sum of train_size and val_size is equal to the length of the dataset
if train_size + val_size != len(subset):
    raise ValueError("Sum of train_size and val_size does not equal the length of the dataset!")

# Split the dataset into train and validation sets
train_dataset, val_dataset = torch.utils.data.random_split(subset, [train_size, val_size])
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

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
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Initialize the CNN model
model = CNN()

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Set the number of epochs
num_epochs = 20

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

    print(f'Epochs: {epoch + 1:5d} | Loss: {epoch_loss:.4f}')
