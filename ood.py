import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.covariance import EmpiricalCovariance
from tqdm.auto import tqdm
import json
import os
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


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
svhn_dataset  = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
ood_loader   = DataLoader(svhn_dataset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default='outputs/resnet18_cifar100.pth')
args = parser.parse_args()

model = ResNet18CIFAR(num_classes=100).to(device)
model.load_state_dict(torch.load(args.checkpoint, map_location=device))
model.eval()
print(f"Loaded checkpoint from {args.checkpoint}")
os.makedirs('outputs', exist_ok=True)


@torch.no_grad()
def extract_features_and_logits(model, loader):
    model.eval()
    all_features, all_logits, all_labels = [], [], []
    for images, labels in tqdm(loader, desc="Extracting"):
        images = images.to(device)
        features, logits = model.get_features_and_logits(images)
        all_features.append(features.cpu())
        all_logits.append(logits.cpu())
        all_labels.append(labels)
    return torch.cat(all_features), torch.cat(all_logits), torch.cat(all_labels)


print("Extracting ID train features...")
train_features, train_logits, train_labels = extract_features_and_logits(model, train_loader)
print("Extracting ID test features...")
test_features, test_logits, test_labels = extract_features_and_logits(model, test_loader)
print("Extracting OOD features...")
ood_features, ood_logits, ood_labels = extract_features_and_logits(model, ood_loader)


def compute_ood_metrics(id_scores, ood_scores):
    labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])
    auroc = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.argmin(np.abs(tpr - 0.95))
    return auroc, fpr[idx]


def score_msp(logits):
    return F.softmax(logits, dim=1).max(dim=1).values.numpy()

def score_max_logit(logits):
    return logits.max(dim=1).values.numpy()

def score_energy(logits, T=1.0):
    return (T * torch.logsumexp(logits / T, dim=1)).numpy()


def compute_class_stats(features, labels, num_classes=100):
    class_means, centered_features = [], []
    for c in range(num_classes):
        mask = (labels == c)
        class_feats = features[mask]
        mu_c = class_feats.mean(dim=0)
        class_means.append(mu_c)
        centered_features.append(class_feats - mu_c)
    class_means = torch.stack(class_means)
    all_centered = torch.cat(centered_features, dim=0)
    cov = EmpiricalCovariance().fit(all_centered.numpy())
    precision = torch.tensor(cov.precision_, dtype=torch.float32)
    return class_means, torch.tensor(cov.covariance_, dtype=torch.float32), precision


def score_mahalanobis(features, class_means, precision):
    diff = features.unsqueeze(1) - class_means.unsqueeze(0)
    maha = torch.einsum('ncd,dd,ncd->nc', diff, precision, diff)
    return -maha.min(dim=1).values.numpy()


def compute_vim_score(features, logits, train_features, class_means, alpha=1.0, num_dims=None):
    D = features.shape[1]
    if num_dims is None:
        num_dims = min(class_means.shape[0], D) - 1
    global_mean = train_features.mean(dim=0)
    centered_train = train_features - global_mean
    U, S, Vt = torch.linalg.svd(centered_train, full_matrices=False)
    P = Vt[:num_dims].T
    centered_test = features - global_mean
    projected = centered_test @ P @ P.T
    residual = torch.norm(centered_test - projected, dim=1)
    energy = torch.logsumexp(logits, dim=1)
    return (energy - alpha * residual).numpy()


def compute_neco_score(features, train_features, num_dims=99, logits=None, rescale=True):
    global_mean = train_features.mean(dim=0)
    centered_train = train_features - global_mean
    centered_test = features - global_mean
    U, S, Vt = torch.linalg.svd(centered_train, full_matrices=False)
    P = Vt[:num_dims].T
    proj_norm = (centered_test @ P).norm(dim=1)
    feat_norm = centered_test.norm(dim=1)
    neco = proj_norm / (feat_norm + 1e-8)
    if rescale and logits is not None:
        neco = neco * logits.max(dim=1).values
    return neco.numpy()


print("\nComputing class statistics...")
class_means, covariance, precision = compute_class_stats(train_features, train_labels)

print("Computing OOD scores...")
msp_id, msp_ood = score_msp(test_logits), score_msp(ood_logits)
ml_id, ml_ood = score_max_logit(test_logits), score_max_logit(ood_logits)
energy_id, energy_ood = score_energy(test_logits), score_energy(ood_logits)
maha_id = score_mahalanobis(test_features, class_means, precision)
maha_ood = score_mahalanobis(ood_features, class_means, precision)
vim_id = compute_vim_score(test_features, test_logits, train_features, class_means)
vim_ood = compute_vim_score(ood_features, ood_logits, train_features, class_means)
neco_id_raw = compute_neco_score(test_features, train_features, rescale=False)
neco_ood_raw = compute_neco_score(ood_features, train_features, rescale=False)
neco_id = compute_neco_score(test_features, train_features, logits=test_logits)
neco_ood = compute_neco_score(ood_features, train_features, logits=ood_logits)

all_methods = {
    'MSP':             (msp_id, msp_ood),
    'Max Logit':       (ml_id, ml_ood),
    'Energy':          (energy_id, energy_ood),
    'Mahalanobis':     (maha_id, maha_ood),
    'ViM':             (vim_id, vim_ood),
    'NECO (raw)':      (neco_id_raw, neco_ood_raw),
    'NECO (rescaled)': (neco_id, neco_ood),
}

print(f"\n{'Method':<20} {'AUROC':>8} {'FPR@95':>8}")
print("=" * 38)
all_results = {}
for name, (id_s, ood_s) in all_methods.items():
    auroc, fpr95 = compute_ood_metrics(id_s, ood_s)
    all_results[name] = (auroc, fpr95)
    print(f"{name:<20} {auroc:>8.4f} {fpr95:>8.4f}")

# Training curves
if os.path.exists('outputs/history.json'):
    with open('outputs/history.json') as f:
        history = json.load(f)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(history['train_loss'], label='Train'); axes[0].plot(history['test_loss'], label='Test')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(history['train_acc'], label='Train'); axes[1].plot(history['test_acc'], label='Test')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy'); axes[1].set_title('Accuracy'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[2].plot(history['lr']); axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('LR'); axes[2].set_title('LR Schedule'); axes[2].grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig('outputs/training_curves.png', dpi=150, bbox_inches='tight'); plt.close()
    print("Saved outputs/training_curves.png")

# Score distributions
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
axes = axes.flatten()
for idx, (name, (id_s, ood_s)) in enumerate(all_methods.items()):
    ax = axes[idx]
    ax.hist(id_s, bins=80, alpha=0.6, density=True, label='ID', color='steelblue')
    ax.hist(ood_s, bins=80, alpha=0.6, density=True, label='OOD', color='coral')
    ax.set_title(f"{name} (AUROC={all_results[name][0]:.4f})")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
axes[-1].axis('off')
plt.suptitle('OOD Score Distributions: CIFAR-100 vs SVHN', fontsize=14)
plt.tight_layout(); plt.savefig('outputs/ood_score_distributions.png', dpi=150, bbox_inches='tight'); plt.close()
print("Saved outputs/ood_score_distributions.png")

# ROC curves
fig, ax = plt.subplots(figsize=(8, 6))
for name, (id_s, ood_s) in all_methods.items():
    labels = np.concatenate([np.ones(len(id_s)), np.zeros(len(ood_s))])
    scores = np.concatenate([id_s, ood_s])
    fpr, tpr, _ = roc_curve(labels, scores)
    ax.plot(fpr, tpr, label=f'{name} ({all_results[name][0]:.4f})', linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title('ROC Curves'); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('outputs/roc_curves.png', dpi=150, bbox_inches='tight'); plt.close()
print("Saved outputs/roc_curves.png")

# Bar comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
names = list(all_results.keys())
aurocs = [all_results[n][0] for n in names]
fpr95s = [all_results[n][1] for n in names]
colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
ax = axes[0]
bars = ax.barh(names, aurocs, color=colors); ax.set_xlabel('AUROC'); ax.set_title('AUROC (Higher = Better)'); ax.set_xlim(0.5, 1.0)
for bar, v in zip(bars, aurocs): ax.text(v + 0.005, bar.get_y() + bar.get_height()/2, f'{v:.4f}', va='center', fontsize=9)
ax.grid(True, alpha=0.3, axis='x')
ax = axes[1]
bars = ax.barh(names, fpr95s, color=colors); ax.set_xlabel('FPR@95'); ax.set_title('FPR@95 (Lower = Better)')
for bar, v in zip(bars, fpr95s): ax.text(v + 0.005, bar.get_y() + bar.get_height()/2, f'{v:.4f}', va='center', fontsize=9)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout(); plt.savefig('outputs/final_comparison.png', dpi=150, bbox_inches='tight'); plt.close()
print("Saved outputs/final_comparison.png")

print("\nDone! All OOD outputs saved to outputs/")
