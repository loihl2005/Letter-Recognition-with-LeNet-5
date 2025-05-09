import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random

# 1. Cấu hình
DATA_DIR = "D:/PPNCKH/EMNIST"   # Đường dẫn thư mục dữ liệu
MODEL_NAME = "lenet5"           # Chọn: "lenet5", "resnet50", "mobilenet_v2", etc.
PRETRAINED = True               # Chỉ áp dụng cho các model torchvision
BATCH_SIZE = 32
NUM_EPOCHS = 5            
LR = 0.001                      # Learning rate
EARLY_STOPPING_PATIENCE = 3
NUM_CLASSES = 26
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Định nghĩa LeNet-5
class LeNet5(nn.Module):
    def __init__(self, num_classes=26):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 3. Transform phù hợp
if MODEL_NAME == 'lenet5':
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.rot90(x, k=1, dims=[1,2])),  # xoay 90° 
        transforms.Lambda(lambda x: torch.flip(x, dims=[1])),          # lật ngang
        transforms.Normalize((0.5,), (0.5,))    
    ])
else:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.rot90(x, k=1, dims=[1,2])),  # xoay 90° 
        transforms.Lambda(lambda x: torch.flip(x, dims=[1])),          # lật ngang
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# 4. Load dataset
train_folder = os.path.join(DATA_DIR, 'letters-training')
test_folder  = os.path.join(DATA_DIR, 'letters-testing')
train_dataset = datasets.ImageFolder(train_folder, transform=transform)
test_dataset  = datasets.ImageFolder(test_folder, transform=transform)

# Tách tập validation từ train
train_size = int(0.8 * len(train_dataset))
val_size   = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 5. Hàm khởi tạo model chung

def get_model(model_name, num_classes, pretrained=True):
    if model_name == 'lenet5':
        return LeNet5(num_classes)
    # Các model pretrained từ torchvision
    try:
        model = getattr(models, model_name)(pretrained=pretrained)
    except AttributeError:
        raise ValueError(f"Model {model_name} không hỗ trợ.")
    # Điều chỉnh lớp cuối cho phù hợp số class
    if model_name.startswith('resnet') or model_name.startswith("vgg"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'mobilenet_v2':
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name.startswith('densenet'):
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == 'inception_v3':
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'alexnet':
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name.startswith('efficientnet'):
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Không biết cách tinh chỉnh model {model_name}!")
    return model

model = get_model(MODEL_NAME, NUM_CLASSES, PRETRAINED).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# 6. Training với Early Stopping
best_val_loss = float('inf')
pati_counter = 0

for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    model.train()
    running_loss = 0.0
    for imgs, labels in tqdm(train_loader):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_train = running_loss / len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            val_loss += criterion(outputs, labels).item()
    avg_val = val_loss / len(val_loader)

    print(f"Train Loss: {avg_train:.4f}, Val Loss: {avg_val:.4f}")
    if avg_val < best_val_loss:
        best_val_loss = avg_val
        pati_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        pati_counter += 1
        if pati_counter >= EARLY_STOPPING_PATIENCE:
            print("Early stopping.")
            break

# 7. Load và đánh giá trên tập test
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for imgs, labels in tqdm(test_loader):
        imgs = imgs.to(DEVICE)
        outs = model(imgs)
        preds = outs.argmax(dim=1)
        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())

acc = accuracy_score(y_true, y_pred)
print(f"Test Accuracy: {acc*100:.2f}%")

# 8. Vẽ confusion matrix với A-Z labels
def int_to_char(i):
    return chr(i + ord('A'))
labels_char = [int_to_char(i) for i in range(NUM_CLASSES)]

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels_char, yticklabels=labels_char)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# 9. Hiển thị mẫu đúng/sai với A-Z
sample_size = 5  # Số lượng mẫu đúng và sai muốn hiển thị

correct_samples = [i for i in range(len(y_true)) if y_true[i] == y_pred[i]]
incorrect_samples = [i for i in range(len(y_true)) if y_true[i] != y_pred[i]]
correct_samples = random.sample(correct_samples, sample_size)
incorrect_samples = random.sample(incorrect_samples, sample_size)

fig, axes = plt.subplots(2, sample_size, figsize=(sample_size*2.5, sample_size*2.5))

def plot_digit(ax, idx):
    img_tensor, _ = test_loader.dataset[idx]
    img = img_tensor.cpu().numpy().transpose(1,2,0).squeeze()  # (H,W)
    ax.imshow(img, cmap='gray', interpolation='nearest')
    ax.axis('off')

for i, idx in enumerate(correct_samples):
    ax = axes[0, i]
    plot_digit(ax, idx)
    ax.set_title(f"True:{int_to_char(y_true[idx])}, Pred:{int_to_char(y_pred[idx])}", pad=4)

for i, idx in enumerate(incorrect_samples):
    ax = axes[1, i]
    plot_digit(ax, idx)
    ax.set_title(f"True:{int_to_char(y_true[idx])}, Pred:{int_to_char(y_pred[idx])}", pad=4)

plt.tight_layout()
plt.show()


