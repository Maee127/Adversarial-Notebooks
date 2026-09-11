# =============================================================
# Adversarial Attacks Series — Note 07
# Knowing the Weights: Four Bayesian Lenses on Uncertainty
# =============================================================
#
# Series:  Humble Model / Bayesian Uncertainty
# Dataset: CIFAR-10
# Model:   CNN (same backbone as Essay #6)
#
# Notebook structure:
#   Part A: Imports and Setup
#   Part B: Dataset Loading (CIFAR-10, held-out split)
#   Part C: Shared Backbone Architecture
#   Part D: Baseline Model (load or train)
#   Part E: Method 1 — Monte Carlo Dropout
#   Part F: Method 2 — Deep Ensembles
#   Part G: Method 3 — SWAG
#   Part H: Method 4 — Variational Inference
#   Part I: Evaluation Protocol
#   Part J: Head-to-Head Comparison
#   Part K: Summary and Outputs
# =============================================================

# -------------------------------------------------------------
# Part A: Imports and Setup
# -------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import random
import json

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# -------------------------------------------------------------
# Part B: Dataset Loading (CIFAR-10) — with held-out split
# -------------------------------------------------------------

cifar_transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

cifar_transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=cifar_transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=cifar_transform_test)

# Held-out split: 20% calibration, 80% evaluation
calib_size = int(0.2 * len(test_dataset))
eval_size = len(test_dataset) - calib_size
calib_dataset, eval_dataset = random_split(test_dataset, [calib_size, eval_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
calib_loader = DataLoader(calib_dataset, batch_size=64, shuffle=False, num_workers=0)
eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False, num_workers=0)

print(f"Training samples: {len(train_dataset):,}")
print(f"Calibration samples: {len(calib_dataset):,}")
print(f"Evaluation samples: {len(eval_dataset):,}")

# Per-channel valid normalized range
CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
CIFAR_STD = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)
CIFAR_MIN = ((0 - CIFAR_MEAN) / CIFAR_STD).to(DEVICE)
CIFAR_MAX = ((1 - CIFAR_MEAN) / CIFAR_STD).to(DEVICE)

def clamp_valid(x):
    return torch.max(torch.min(x, CIFAR_MAX), CIFAR_MIN)

# -------------------------------------------------------------
# Part C: Shared Backbone Architecture
# -------------------------------------------------------------

class BackboneCNN(nn.Module):
    """Shared backbone for all Bayesian methods."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return x

class StandardCNN(nn.Module):
    """Standard CNN for baseline."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = BackboneCNN()
        self.fc_out = nn.Linear(256, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.fc_out(features)

# -------------------------------------------------------------
# Part D: Baseline Model (load or train)
# -------------------------------------------------------------

def train_model(model, train_loader, epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch} — avg loss: {avg_loss:.4f}")
    return model

def evaluate_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total

baseline_model = StandardCNN().to(DEVICE)

if os.path.exists('checkpoint_baseline_cifar10.pth'):
    checkpoint = torch.load('checkpoint_baseline_cifar10.pth', map_location=DEVICE)
    baseline_model.load_state_dict(checkpoint['model_state_dict'])
    print("Loaded baseline model from checkpoint")
else:
    print("Training baseline model from scratch...")
    baseline_model = train_model(baseline_model, train_loader, epochs=20)
    torch.save({'model_state_dict': baseline_model.state_dict()}, 'checkpoint_baseline_cifar10.pth')
    print("Saved baseline model checkpoint")

clean_acc = evaluate_accuracy(baseline_model, eval_loader)
print(f"Baseline clean test accuracy: {clean_acc:.2f}%")

# -------------------------------------------------------------
# Part E: Method 1 — Monte Carlo Dropout
# -------------------------------------------------------------

class DropoutCNN(nn.Module):
    """CNN with dropout in the backbone for MC Dropout inference."""
    def __init__(self, num_classes=10, dropout_rate=0.3):
        super().__init__()
        self.backbone = BackboneCNN()
        self.fc_out = nn.Linear(256, num_classes)
        # Replace the backbone's dropout with the desired rate
        self.backbone.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        features = self.backbone(x)
        return self.fc_out(features)

mc_model = DropoutCNN(dropout_rate=0.3).to(DEVICE)

if os.path.exists('checkpoint_mc_dropout.pth'):
    checkpoint = torch.load('checkpoint_mc_dropout.pth', map_location=DEVICE)
    mc_model.load_state_dict(checkpoint['model_state_dict'])
    print("Loaded MC Dropout model from checkpoint")
else:
    print("Training MC Dropout model...")
    mc_model = train_model(mc_model, train_loader, epochs=20)
    torch.save({'model_state_dict': mc_model.state_dict()}, 'checkpoint_mc_dropout.pth')
    print("Saved MC Dropout model checkpoint")

def mc_dropout_predict(model, image, num_passes=50):
    """Run MC Dropout inference."""
    model.train()  # enable dropout
    predictions = []
    for _ in range(num_passes):
        with torch.no_grad():
            pred = torch.softmax(model(image), dim=1)
            predictions.append(pred)
    model.eval()
    predictions = torch.stack(predictions)  # [N, batch, classes]
    mean_pred = predictions.mean(dim=0)
    variance = predictions.var(dim=0)
    return mean_pred, variance

# Sanity check
test_image, test_label = eval_dataset[0]
test_image = test_image.unsqueeze(0).to(DEVICE)
mean_pred, variance = mc_dropout_predict(mc_model, test_image, num_passes=50)
print(f"\nMC Dropout sanity check:")
print(f"  True label: {test_label}")
print(f"  Predicted: {mean_pred.argmax().item()}")
print(f"  Confidence: {mean_pred.max().item():.4f}")
print(f"  Variance (max): {variance.max().item():.6f}")

# -------------------------------------------------------------
# Part F: Method 2 — Deep Ensembles
# -------------------------------------------------------------

NUM_ENSEMBLE = 5
ensemble_models = []

for i in range(NUM_ENSEMBLE):
    ckpt_path = f'checkpoint_ensemble_{i}.pth'
    model = StandardCNN().to(DEVICE)
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        ensemble_models.append(model)
        print(f"Loaded ensemble member {i}")
    else:
        print(f"Training ensemble member {i}...")
        set_seed(i)
        model = train_model(model, train_loader, epochs=20)
        torch.save({'model_state_dict': model.state_dict()}, ckpt_path)
        ensemble_models.append(model)
        print(f"Saved ensemble member {i}")

def ensemble_predict(models, image):
    """Run ensemble inference."""
    predictions = []
    for model in models:
        model.eval()
        with torch.no_grad():
            pred = torch.softmax(model(image), dim=1)
            predictions.append(pred)
    predictions = torch.stack(predictions)
    mean_pred = predictions.mean(dim=0)
    variance = predictions.var(dim=0)
    return mean_pred, variance

# Sanity check
mean_pred_ens, variance_ens = ensemble_predict(ensemble_models, test_image)
print(f"\nDeep Ensemble sanity check:")
print(f"  True label: {test_label}")
print(f"  Predicted: {mean_pred_ens.argmax().item()}")
print(f"  Confidence: {mean_pred_ens.max().item():.4f}")
print(f"  Variance (max): {variance_ens.max().item():.6f}")

# -------------------------------------------------------------
# Part G: Method 3 — SWAG
# -------------------------------------------------------------

class SWAG:
    """
    Stochastic Weight Averaging — Gaussian.
    Fits a Gaussian to the trajectory of SGD iterates.
    """
    def __init__(self, model, max_models=20, num_samples=30):
        self.model = model
        self.max_models = max_models
        self.num_samples = num_samples
        self.means = None
        self.sq_means = None
        self.deviations = None
        self.num_collected = 0

    def collect(self):
        """Collect current weights."""
        params = [p.data.clone().flatten() for p in self.model.parameters()]
        flat_params = torch.cat(params)
        if self.means is None:
            self.means = flat_params.clone()
            self.sq_means = flat_params.pow(2).clone()
        else:
            self.means = (self.means * self.num_collected + flat_params) / (self.num_collected + 1)
            self.sq_means = (self.sq_means * self.num_collected + flat_params.pow(2)) / (self.num_collected + 1)
        self.num_collected += 1

    def fit(self):
        """Fit the Gaussian: compute mean and covariance (diagonal approximation)."""
        self.means = self.means / 1.0  # already averaged
        variance = self.sq_means - self.means.pow(2)
        self.deviations = torch.sqrt(torch.clamp(variance, min=1e-6))

    def sample(self, num_samples=None):
        """Sample weights from the fitted Gaussian."""
        if num_samples is None:
            num_samples = self.num_samples
        samples = []
        for _ in range(num_samples):
            sampled = self.means + self.deviations * torch.randn_like(self.means)
            samples.append(sampled)
        return samples

    def load_sample(self, sampled_flat):
        """Load a flattened sample back into the model."""
        offset = 0
        for p in self.model.parameters():
            numel = p.numel()
            p.data.copy_(sampled_flat[offset:offset+numel].view(p.shape))
            offset += numel

# Train SWAG model
swag_model = StandardCNN().to(DEVICE)
swag = SWAG(swag_model, max_models=20, num_samples=30)

if os.path.exists('checkpoint_swag.pth'):
    checkpoint = torch.load('checkpoint_swag.pth', map_location=DEVICE)
    swag_model.load_state_dict(checkpoint['model_state_dict'])
    swag.means = checkpoint['swag_means']
    swag.deviations = checkpoint['swag_deviations']
    print("Loaded SWAG model from checkpoint")
else:
    print("Training SWAG model...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(swag_model.parameters(), lr=0.001)
    for epoch in range(20):
        swag_model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = swag_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch} — avg loss: {running_loss/len(train_loader):.4f}")
        # Collect weights at the end of each epoch
        swag.collect()
    swag.fit()
    torch.save({
        'model_state_dict': swag_model.state_dict(),
        'swag_means': swag.means,
        'swag_deviations': swag.deviations,
    }, 'checkpoint_swag.pth')
    print("Saved SWAG model checkpoint")

def swag_predict(swag, image, num_samples=30):
    """Run SWAG inference by sampling weights."""
    predictions = []
    for _ in range(num_samples):
        sampled = swag.means + swag.deviations * torch.randn_like(swag.means)
        swag.load_sample(sampled)
        swag.model.eval()
        with torch.no_grad():
            pred = torch.softmax(swag.model(image), dim=1)
            predictions.append(pred)
    # Restore mean weights
    swag.load_sample(swag.means)
    predictions = torch.stack(predictions)
    mean_pred = predictions.mean(dim=0)
    variance = predictions.var(dim=0)
    return mean_pred, variance

# Sanity check
mean_pred_swag, variance_swag = swag_predict(swag, test_image, num_samples=30)
print(f"\nSWAG sanity check:")
print(f"  True label: {test_label}")
print(f"  Predicted: {mean_pred_swag.argmax().item()}")
print(f"  Confidence: {mean_pred_swag.max().item():.4f}")
print(f"  Variance (max): {variance_swag.max().item():.6f}")

# -------------------------------------------------------------
# Part H: Method 4 — Variational Inference (Bayes by Backprop)
# -------------------------------------------------------------

class BayesianLinear(nn.Module):
    """Bayesian linear layer with mean and log-variance parameters."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mean = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_logvar = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias_mean = nn.Parameter(torch.zeros(out_features))
        self.bias_logvar = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        weight_std = torch.exp(0.5 * self.weight_logvar)
        bias_std = torch.exp(0.5 * self.bias_logvar)
        weight = self.weight_mean + weight_std * torch.randn_like(weight_std)
        bias = self.bias_mean + bias_std * torch.randn_like(bias_std)
        return F.linear(x, weight, bias)

    def kl_divergence(self):
        """KL divergence from the variational posterior to the prior N(0, I)."""
        weight_kl = 0.5 * (self.weight_mean.pow(2) + self.weight_logvar.exp() - self.weight_logvar - 1).sum()
        bias_kl = 0.5 * (self.bias_mean.pow(2) + self.bias_logvar.exp() - self.bias_logvar - 1).sum()
        return weight_kl + bias_kl

class BayesianCNN(nn.Module):
    """CNN with Bayesian layers for Variational Inference."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = BayesianLinear(128 * 4 * 4, 256)
        self.fc2 = BayesianLinear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def kl_divergence(self):
        return self.fc1.kl_divergence() + self.fc2.kl_divergence()

vi_model = BayesianCNN().to(DEVICE)

if os.path.exists('checkpoint_vi.pth'):
    checkpoint = torch.load('checkpoint_vi.pth', map_location=DEVICE)
    vi_model.load_state_dict(checkpoint['model_state_dict'])
    print("Loaded VI model from checkpoint")
else:
    print("Training VI model...")
    optimizer = optim.Adam(vi_model.parameters(), lr=0.001)
    num_batches = len(train_loader)
    for epoch in range(20):
        vi_model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = vi_model(images)
            ce_loss = F.cross_entropy(outputs, labels)
            kl_loss = vi_model.kl_divergence() / num_batches
            loss = ce_loss + kl_loss * 0.001  # scale the KL term
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch} — avg loss: {running_loss/len(train_loader):.4f}")
    torch.save({'model_state_dict': vi_model.state_dict()}, 'checkpoint_vi.pth')
    print("Saved VI model checkpoint")

def vi_predict(model, image, num_samples=30):
    """Run VI inference by sampling weights."""
    model.eval()
    predictions = []
    for _ in range(num_samples):
        with torch.no_grad():
            pred = torch.softmax(model(image), dim=1)
            predictions.append(pred)
    predictions = torch.stack(predictions)
    mean_pred = predictions.mean(dim=0)
    variance = predictions.var(dim=0)
    return mean_pred, variance

# Sanity check
mean_pred_vi, variance_vi = vi_predict(vi_model, test_image, num_samples=30)
print(f"\nVI sanity check:")
print(f"  True label: {test_label}")
print(f"  Predicted: {mean_pred_vi.argmax().item()}")
print(f"  Confidence: {mean_pred_vi.max().item():.4f}")
print(f"  Variance (max): {variance_vi.max().item():.6f}")

# -------------------------------------------------------------
# Part I: Evaluation Protocol
# -------------------------------------------------------------

def fgsm_attack(model, images, labels, epsilon=0.03):
    model.eval()
    images = images.clone().detach().to(DEVICE)
    images.requires_grad = True
    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels.to(DEVICE))
    model.zero_grad()
    loss.backward()
    perturbed = images + epsilon * images.grad.sign()
    return clamp_valid(perturbed).detach()

def pgd_attack(model, images, labels, epsilon=0.03, step_size=0.007, num_steps=40):
    model.eval()
    images_adv = images.clone().detach().to(DEVICE)
    for _ in range(num_steps):
        images_adv.requires_grad = True
        outputs = model(images_adv)
        loss = nn.CrossEntropyLoss()(outputs, labels.to(DEVICE))
        model.zero_grad()
        loss.backward()
        grad_sign = images_adv.grad.sign()
        images_adv = images_adv + step_size * grad_sign
        perturbation = torch.clamp(images_adv - images, -epsilon, epsilon)
        images_adv = torch.clamp(images + perturbation, 0, 1).detach()
    return images_adv

def evaluate_with_deferral(predict_fn, loader, threshold, attack_type=None, attack_params=None):
    """
    Evaluate a prediction function with deferral.
    predict_fn should return (mean_pred, variance) for a batch.
    """
    correct = 0
    total = 0
    deferred = 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        if attack_type == 'fgsm':
            images = fgsm_attack(baseline_model, images, labels, **attack_params)
        elif attack_type == 'pgd':
            images = pgd_attack(baseline_model, images, labels, **attack_params)
        for i in range(images.size(0)):
            img = images[i].unsqueeze(0)
            mean_pred, variance = predict_fn(img)
            uncertainty = variance.max().item()
            if uncertainty > threshold:
                deferred += 1
            else:
                pred = mean_pred.argmax().item()
                if pred == labels[i].item():
                    correct += 1
                total += 1
    accuracy = 100 * correct / total if total > 0 else 0.0
    deferral_rate = 100 * deferred / (total + deferred) if (total + deferred) > 0 else 0.0
    risk = 100 - accuracy if total > 0 else 0.0
    return {'accuracy': accuracy, 'deferral_rate': deferral_rate, 'risk': risk,
            'coverage': 100 - deferral_rate, 'total': total + deferred}

# -------------------------------------------------------------
# Part J: Head-to-Head Comparison
# -------------------------------------------------------------

print("\n" + "="*55)
print("Bayesian Methods — Placeholder for Results")
print("="*55)
print("""
To be filled in after you run the evaluation for each method:

| Method              | Clean Coverage | Clean Risk | Adv Coverage | Adv Risk | Cost   |
|---------------------|----------------|------------|--------------|----------|--------|
| MC Dropout          | TBD            | TBD        | TBD          | TBD      | Low    |
| Deep Ensembles      | TBD            | TBD        | TBD          | TBD      | Medium |
| SWAG                | TBD            | TBD        | TBD          | TBD      | Medium |
| Variational Infer.  | TBD            | TBD        | TBD          | TBD      | High   |

Comparison against Essay #6 architecture-native signals:
| Method              | Type              | Adv Risk | Cost   |
|---------------------|-------------------|----------|--------|
| Multi-head          | Architecture      | 18.92%   | Medium |
| Evidential          | Architecture      | 20.64%   | Medium |
| MC Dropout          | Bayesian          | TBD      | Low    |
| Deep Ensembles      | Bayesian          | TBD      | Medium |
| SWAG                | Bayesian          | TBD      | Medium |
| Variational Infer.  | Bayesian          | TBD      | High   |
""")

# -------------------------------------------------------------
# Part K: Summary and Outputs
# -------------------------------------------------------------

print("\n" + "="*55)
print("Experiment Summary")
print("="*55)

print(f"""
Baseline:
  - Clean accuracy: {clean_acc:.2f}%

MC Dropout:
  - Sanity prediction: {mean_pred.argmax().item()}
  - Sanity confidence: {mean_pred.max().item():.4f}
  - Sanity variance: {variance.max().item():.6f}

Deep Ensembles:
  - Sanity prediction: {mean_pred_ens.argmax().item()}
  - Sanity confidence: {mean_pred_ens.max().item():.4f}
  - Sanity variance: {variance_ens.max().item():.6f}

SWAG:
  - Sanity prediction: {mean_pred_swag.argmax().item()}
  - Sanity confidence: {mean_pred_swag.max().item():.4f}
  - Sanity variance: {variance_swag.max().item():.6f}

Variational Inference:
  - Sanity prediction: {mean_pred_vi.argmax().item()}
  - Sanity confidence: {mean_pred_vi.max().item():.4f}
  - Sanity variance: {variance_vi.max().item():.6f}

Output files:
  - checkpoint_baseline_cifar10.pth
  - checkpoint_mc_dropout.pth
  - checkpoint_ensemble_*.pth
  - checkpoint_swag.pth
  - checkpoint_vi.pth
""")

print("\nNotebook complete!")