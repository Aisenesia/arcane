import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
import os
import cv2

# Experimental code

# Define paths
dataset_path = 'datasets/train/'

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.image_files = []
        self.label_files = []
        
        for class_dir in os.listdir(images_dir):
            class_image_dir = os.path.join(images_dir, class_dir)
            class_label_dir = os.path.join(labels_dir, class_dir)
            for img_file in os.listdir(class_image_dir):
                self.image_files.append(os.path.join(class_image_dir, img_file))
                self.label_files.append(os.path.join(class_label_dir, img_file.replace('.jpg', '.txt')))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label_path = self.label_files[idx]
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        
        with open(label_path, 'r') as f:
            label = int(f.readline().strip().split()[0])
            if label < 0 or label >= 20:
                raise ValueError(f"Label {label} out of range for file {label_path}")
        
        return image, label

# Data augmentation and preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = CustomDataset(images_dir=os.path.join(dataset_path, 'images'), labels_dir=os.path.join(dataset_path, 'labels'), transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define the model
class DiceCNN(nn.Module):
    def __init__(self):
        super(DiceCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 20)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = DiceCNN()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 20
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

# Save the model
torch.save(model.state_dict(), 'dice_model.pth')
