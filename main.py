import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from model import TinyVGG

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
batch_size = 32
num_epochs = 10
learning_rate = 0.01

# Data transformation
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Download and split datasets
train_data = datasets.FashionMNIST(root='data', train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST(root='data', train=False, download=True, transform=transform)

# Split train into train and validation
train_size = int(0.8 * len(train_data))
val_size = len(train_data) - train_size
train_data, val_data = random_split(train_data, [train_size, val_size])

# Create dataloaders
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Initialize model, loss, optimizer
model = TinyVGG().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
train_losses = []
train_accs = []
val_losses = []
val_accs = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * y.size(0)
        preds = y_pred.argmax(dim=1)
        train_correct += (preds == y).sum().item()
        train_total += y.size(0)

    train_loss = train_loss / train_total
    train_acc = train_correct / train_total
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            val_loss += loss.item() * y.size(0)
            preds = y_pred.argmax(dim=1)
            val_correct += (preds == y).sum().item()
            val_total += y.size(0)

    val_loss = val_loss / val_total
    val_acc = val_correct / val_total
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}\n")

# Plot loss and accuracy curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('training_curves.png')
plt.close()

# Test evaluation and confusion matrix
test_preds = []
test_labels = []
model.eval()
with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        test_preds.extend(y_pred.argmax(dim=1).cpu().numpy())
        test_labels.extend(y.cpu().numpy())

test_acc = (np.array(test_preds) == np.array(test_labels)).mean()
print(f"Test Accuracy: {test_acc:.4f}")

cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()



# Visualize feature maps from conv_block_1
# Select a random test image
random_idx = torch.randint(0, len(test_data), (1,)).item()
image, label = test_data[random_idx]
image_tensor = image.unsqueeze(0).to(device)  # Add batch dimension [1, 1, 28, 28]

# Extract feature maps from conv_block_1
model.eval()
with torch.no_grad():
    features = model.conv_block1(image_tensor)  # Output shape: [1, 10, 14, 14]

# Remove batch dimension and move to CPU
features = features.squeeze(0).cpu().numpy()  # Shape [10, 14, 14]

# Plot original image and feature maps
fig, axs = plt.subplots(3, 5, figsize=(15, 8))
axs[0, 0].imshow(image.squeeze(), cmap='gray')
axs[0, 0].set_title(f'Original (Label: {label})')
axs[0, 0].axis('off')

# Hide unused subplots in the first row
for i in range(1, 5):
    axs[0, i].axis('off')

# Plot feature maps
for i in range(10):
    row = 1 + i // 5
    col = i % 5
    axs[row, col].imshow(features[i], cmap='gray')
    axs[row, col].set_title(f'Channel {i}')
    axs[row, col].axis('off')

# Hide unused subplots in the last row
for i in range(10, 15):
    row = 2
    col = i % 5
    axs[row, col].axis('off')

plt.tight_layout()
plt.savefig('feature_maps.png')
plt.close()

print("Feature maps saved to feature_maps.png")