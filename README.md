# Out-of-Distribution Detection & Neural Collapse

Training a ResNet-18 on CIFAR-100, then studying two closely related questions: can we detect
out-of-distribution inputs from the trained model's representations, and does the network's last
layer spontaneously organize into the symmetric geometric structure predicted by the Neural
Collapse theory?

*Theory of Deep Learning — MVA / ENSTA, February 2026.*
Asma Walha, Ameni Hidouri.

## Setup

ResNet-18 adapted for 32×32 inputs (3×3 first conv, no max-pool), trained on CIFAR-100 for 100
epochs (SGD, momentum 0.9, weight decay 5e-4, cosine annealing from lr=0.1). SVHN is used as
the out-of-distribution dataset throughout.

| Metric | Value |
|---|---|
| Final train accuracy | 99.96% |
| Final test accuracy | 77.13% |
| Parameters | 11.2M |

## Out-of-Distribution Detection

Seven scoring methods compared, from simple output-based scores to feature-geometry-aware
ones, evaluated by AUROC and FPR@95 (CIFAR-100 vs. SVHN):

| Method | Type | AUROC ↑ | FPR@95 ↓ |
|---|---|---|---|
| MSP | Output-based | 0.8508 | 0.6606 |
| Max Logit | Output-based | 0.8807 | 0.6292 |
| Energy | Output-based | 0.8846 | 0.6041 |
| Mahalanobis | Feature-based | 0.7847 | 0.8291 |
| **ViM** | **Hybrid** | **0.9027** | **0.5163** |
| NECO (raw) | Feature-based | 0.8477 | 0.5813 |
| NECO (rescaled) | Hybrid | 0.8834 | 0.6208 |

**ViM** (energy score combined with the residual norm outside the ID feature subspace) gives
the best separation. The progression MSP → Max Logit → Energy shows that moving from
softmax probabilities to raw logits to full LogSumExp aggregation steadily helps; Mahalanobis
underperforms here, likely due to noisy covariance estimation in a 512-dim feature space from
only 500 samples/class.

## Neural Collapse (NC1–NC5)

Neural Collapse describes a symmetric geometric structure that emerges in a network's last layer
once trained to near-zero loss. All four classical properties were confirmed on this model:

| Property | What it measures | Result |
|---|---|---|
| NC1 — Variability collapse | Within-class vs. between-class feature spread | 0.509 |
| NC2 — Simplex ETF | Class means equiangular on a hypersphere | mean cos = −0.0100 (theory: −0.0101) |
| NC3 — Self-duality | Classifier weights align with class means | mean cos = 0.963 |
| NC4 — NCC simplification | Classifier ≈ nearest-class-mean rule | 95.8% agreement, 77.1% vs 76.9% accuracy |
| NC5 — ID/OOD orthogonality | OOD features orthogonal to the ID subspace | OrthoDev = 0.106 |

NC5 — that OOD features land roughly orthogonal to the subspace spanned by ID class means —
is what motivates **NECO**, a Neural-Collapse-inspired OOD score: it measures what fraction of a
sample's feature norm projects onto the ID principal subspace. It performs competitively (AUROC
0.883 rescaled) without ever looking at the OOD data during design, confirming that the same
geometric structure characterizing in-distribution representations also determines where
out-of-distribution features end up.

## Repository contents

- `train.py` — trains the ResNet-18 backbone on CIFAR-100, saves checkpoint + training curves.
- `ood.py` — computes all 7 OOD scores, evaluation metrics, and comparison plots.
- `neural_collapse.py` — computes NC1–NC5 and produces the corresponding figures.
- `requirements.txt`

## Usage

```bash
pip install -r requirements.txt
python train.py                 # trains and saves outputs/resnet18_cifar100.pth
python ood.py --checkpoint outputs/resnet18_cifar100.pth
python neural_collapse.py --checkpoint outputs/resnet18_cifar100.pth
```