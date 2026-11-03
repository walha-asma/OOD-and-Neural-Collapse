import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from tqdm.auto import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
])

train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
test_dataset  = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)


class ResNet18CIFAR(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.backbone = resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(self.feature_dim, num_classes)
        self._features = None
        self.backbone.avgpool.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        self._features = output.flatten(1)

    def forward(self, x):
        return self.backbone(x)

    def get_features_and_logits(self, x):
        logits = self.forward(x)
        return self._features, logits


model = ResNet18CIFAR(num_classes=100).to(device)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

NUM_EPOCHS = 100
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
criterion = nn.CrossEntropyLoss()
history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [], 'lr': []}


def evaluate(model, loader):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += criterion(logits, labels).item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += images.size(0)
    return total_loss / total, correct / total


print("Starting training : ")
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)

    train_loss = running_loss / total
    train_acc = correct / total
    test_loss, test_acc = evaluate(model, test_loader)
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['test_loss'].append(test_loss)
    history['test_acc'].append(test_acc)
    history['lr'].append(scheduler.get_last_lr()[0])
    scheduler.step()

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

print(f"\nFinal Test Accuracy: {history['test_acc'][-1]:.4f}")

os.makedirs('outputs', exist_ok=True)
torch.save(model.state_dict(), 'outputs/resnet18_cifar100.pth')
with open('outputs/history.json', 'w') as f:
    json.dump(history, f)

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].plot(history['train_loss'], label='Train Loss')
axes[0].plot(history['test_loss'], label='Test Loss')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].set_title('Loss Curves')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(history['train_acc'], label='Train Acc')
axes[1].plot(history['test_acc'], label='Test Acc')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy'); axes[1].set_title('Accuracy Curves')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

axes[2].plot(history['lr'])
axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Learning Rate'); axes[2].set_title('LR Schedule')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/training_curves.png', dpi=150, bbox_inches='tight')
plt.close()

print("Saved model, history, and training curves to outputs/")