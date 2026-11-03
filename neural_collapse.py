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
from tqdm.auto import tqdm
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

# NC1: Variability Collapse

def compute_nc1(features, labels, num_classes=100):
    g_mean = features.mean(dim=0)
    Sigma_W = torch.zeros(features.shape[1], features.shape[1])
    c_means = []
    for c in range(num_classes):
        mask = (labels == c)
        feats_c = features[mask]
        mu_c = feats_c.mean(dim=0)
        c_means.append(mu_c)
        centered = feats_c - mu_c
        Sigma_W += (centered.T @ centered) / len(feats_c)
    Sigma_W /= num_classes
    c_means = torch.stack(c_means)
    centered_means = c_means - g_mean
    Sigma_B = (centered_means.T @ centered_means) / num_classes
    Sigma_B_pinv = torch.linalg.pinv(Sigma_B)
    nc1 = torch.trace(Sigma_W @ Sigma_B_pinv).item() / num_classes
    return nc1, Sigma_W, Sigma_B, c_means, g_mean

print("\n Neural Collapse Analysis")
nc1_value, Sigma_W, Sigma_B, class_means, global_mean = compute_nc1(train_features, train_labels)
print(f"NC1 (Variability Collapse) = {nc1_value:.6f}  (lower = more collapsed)")

# NC2: Simplex ETF

def compute_nc2(class_means, global_mean):
    centered = class_means - global_mean
    norms = centered.norm(dim=1, keepdim=True)
    normalized = centered / (norms + 1e-8)
    cos_sim = normalized @ normalized.T
    C = class_means.shape[0]
    mask = ~torch.eye(C, dtype=bool)
    off_diag = cos_sim[mask]
    theoretical_cos = -1.0 / (C - 1)
    return off_diag.mean().item(), off_diag.std().item(), theoretical_cos, (norms.squeeze() / norms.mean()).std().item(), cos_sim

mean_cos, std_cos, theoretical_cos, norm_std, cos_sim_matrix = compute_nc2(class_means, global_mean)
print(f"NC2 (Simplex ETF): mean_cos={mean_cos:.6f}, theoretical={theoretical_cos:.6f}, std={std_cos:.6f}, norm_std={norm_std:.6f}")

# NC3: Self-Duality

W = model.backbone.fc.weight.detach().cpu()
centered_means = class_means - global_mean
W_norm = W / (W.norm(dim=1, keepdim=True) + 1e-8)
M_norm = centered_means / (centered_means.norm(dim=1, keepdim=True) + 1e-8)
nc3_cosines = (W_norm * M_norm).sum(dim=1)
print(f"NC3 (Self-Duality): mean_cos={nc3_cosines.mean().item():.6f}, std={nc3_cosines.std().item():.6f}, min={nc3_cosines.min().item():.6f}")

# NC4: NCC Simplification

pred_classifier = test_logits.argmax(dim=1)
diff = test_features.unsqueeze(1) - class_means.unsqueeze(0)
distances = (diff ** 2).sum(dim=2)
pred_ncc = distances.argmin(dim=1)
agreement = (pred_classifier == pred_ncc).float().mean().item()
acc_cls = (pred_classifier == test_labels).float().mean().item()
acc_ncc = (pred_ncc == test_labels).float().mean().item()
print(f"NC4 (NCC Simplification): agreement={agreement:.4f}, classifier_acc={acc_cls:.4f}, ncc_acc={acc_ncc:.4f}")

# NC1-NC4 Plots

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

subset = 20
im = axes[0].imshow(cos_sim_matrix[:subset, :subset].numpy(), cmap='RdBu_r', vmin=-0.2, vmax=0.2)
axes[0].set_title(f'NC2: Pairwise Cosine (first {subset} classes)')
axes[0].set_xlabel('Class'); axes[0].set_ylabel('Class')
plt.colorbar(im, ax=axes[0], shrink=0.8)

axes[1].hist(nc3_cosines.numpy(), bins=30, color='steelblue', edgecolor='white', alpha=0.8)
axes[1].axvline(nc3_cosines.mean().item(), color='red', linestyle='--', label=f'Mean={nc3_cosines.mean():.3f}')
axes[1].set_title('NC3: cos(w_c, μ_c − μ_G)'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

eigenvalues_W = torch.linalg.eigvalsh(Sigma_W)
eigenvalues_B = torch.linalg.eigvalsh(Sigma_B)
axes[2].semilogy(sorted(eigenvalues_W.numpy(), reverse=True)[:50], label='Σ_W', color='coral')
axes[2].semilogy(sorted(eigenvalues_B.numpy(), reverse=True)[:50], label='Σ_B', color='steelblue')
axes[2].set_title('NC1: Eigenvalue Spectrum'); axes[2].legend(); axes[2].grid(True, alpha=0.3)

plt.tight_layout(); plt.savefig('outputs/neural_collapse_nc1_nc4.png', dpi=150, bbox_inches='tight'); plt.close()
print("Saved outputs/neural_collapse_nc1_nc4.png")

# NC5: ID/OOD Orthogonality

ood_global_mean = ood_features.mean(dim=0)
centered_ood = ood_global_mean - global_mean
centered_id_means = class_means - global_mean
nc5_cosines = F.cosine_similarity(centered_id_means, centered_ood.unsqueeze(0), dim=1)
ortho_dev = nc5_cosines.abs().mean().item()
print(f"NC5 (OOD Orthogonality): OrthoDev={ortho_dev:.6f}, mean_cos={nc5_cosines.mean().item():.6f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(range(len(nc5_cosines)), nc5_cosines.numpy(), color='coral', alpha=0.7, width=1.0)
axes[0].axhline(0, color='black', linewidth=0.5)
axes[0].set_title('NC5: cos per class'); axes[0].set_xlabel('Class Index'); axes[0].grid(True, alpha=0.3)
axes[1].hist(nc5_cosines.abs().numpy(), bins=30, color='coral', edgecolor='white', alpha=0.8)
axes[1].axvline(ortho_dev, color='red', linestyle='--', label=f'Mean |cos|={ortho_dev:.4f}')
axes[1].set_title('NC5: |cos| distribution'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('outputs/neural_collapse_nc5.png', dpi=150, bbox_inches='tight'); plt.close()
print("Saved outputs/neural_collapse_nc5.png")

print("\nDone! All Neural Collapse outputs saved to outputs/")