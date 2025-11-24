import numpy as np
import torch
from torch.utils.data import Dataset
from collections import Counter


def compute_proportions(matrices):
    total_holds = 0
    total_starts = 0
    total_middles = 0
    total_finishes = 0
    total_feet = 0
    for matrix in matrices:
        for line in matrix:
            for element in line:
                total_holds += 1
                if element == 1:
                    total_starts += 1
                if element == 2:
                    total_middles += 1
                if element == 3:
                    total_finishes += 1
                if element == 4:
                    total_feet += 1
    proportions_start = total_starts / total_holds
    proportions_middle = total_middles / total_starts
    proportions_finish = total_finishes / total_feet
    proportions_feet = total_feet / total_finishes
    proportions_unused = 1 - (proportions_start + proportions_middle + proportions_finish + proportions_feet)
    return [proportions_unused, proportions_start, proportions_middle, proportions_finish, proportions_feet]

#Training dataset
class TrainingDataset(Dataset):
    def __init__(self, name, description):
        from src.utils.Utils import load_dataset
        f = load_dataset(name, description)
        matrices = f["matrices"][:]   # shape (N, H, W)

        self.proportions = compute_proportions(matrices)

        self.H = matrices.shape[1]
        self.W = matrices.shape[2]

        # labels entiers 0–4 pour CrossEntropy
        self.labels = torch.tensor(matrices, dtype=torch.long)

        self.x = torch.tensor(matrices, dtype=torch.long)  # labels 0..4 for loss
        self.x_onehot = F.one_hot(self.x, num_classes=5).permute(0, 3, 1, 2).float() # for encoder
        self.x = self.x.unsqueeze(1)

        angles = torch.tensor(f["angles"][:], dtype=torch.float32)
        grades = torch.tensor(f["grades"][:], dtype=torch.float32)

        # stocke min/max utilisés pour normalisation
        self.angles_min = angles.min()
        self.angles_max = angles.max()
        self.grades_min = grades.min()
        self.grades_max = grades.max()

        # normalisation automatique sur [0,1]
        angles = (angles - self.angles_min) / (self.angles_max - self.angles_min + 1e-8)
        grades = (grades - self.grades_min) / (self.grades_max - self.grades_min + 1e-8)

        self.cond = torch.stack([angles, grades], dim=1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x_onehot[i], self.cond[i], self.x[i]

import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, H, W, latent_dim=16):
        super().__init__()

        # Generic CNN
        self.conv = nn.Sequential(
            nn.Conv2d(5, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU()
        )

        # Dimension computation from dataset shape
        with torch.no_grad():
            dummy = torch.zeros(1, 5, H, W)
            out = self.conv(dummy)
            self.flat_dim = out.numel()

        self.fc = nn.Linear(self.flat_dim + 2, 256)
        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)

    def forward(self, x, cond):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        h = torch.cat([h, cond], dim=1)
        h = F.relu(self.fc(h))
        return self.mu(h), self.logvar(h)

class Decoder(nn.Module):
    def __init__(self, H, W, encoder, latent_dim=16):
        super().__init__()

        self.H = H
        self.W = W
        self.latent_dim = latent_dim
        self.encoder = encoder  # getting flat_dim and final CNN shape

        # compute compressed shape automatically
        with torch.no_grad():
            dummy = torch.zeros(1, 5, H, W)
            out = self.encoder.conv(dummy)
            self.shape_after_conv = out.shape  # (1, C, Hc, Wc)
            self.flat_dim = out.numel()

        self.fc = nn.Linear(latent_dim + 2, self.flat_dim)

        C, Hc, Wc = self.shape_after_conv[1:]

        self.C, self.Hc, self.Wc = C, Hc, Wc

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(C, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.ConvTranspose2d(16, 5, 4, stride=2, padding=1)
        )

    def forward(self, z, cond):
        h = torch.cat([z, cond], dim=1)
        h = F.relu(self.fc(h))
        h = h.view(-1, self.C, self.Hc, self.Wc)

        x = self.deconv(h)

        # Ensure exact output size without manual math
        x = F.interpolate(x, size=(self.H, self.W), mode="bilinear", align_corners=False)

        return x


def loss_fn(logits, labels, mu, logvar, weight=None, kl_weight=1.0):
    # logits: (N,5,H,W)  labels: (N,H,W)
    recon = F.cross_entropy(logits, labels, weight=weight, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    sparsity = 0.0005 * logits.abs().mean()
    return recon + kl + sparsity

class VariationalAutoEncoder(nn.Module):
    def __init__(self, H, W, angle_min, angle_max, grades_min, grades_max, latent_dim=16):
        super().__init__()
        self.H = H
        self.W = W
        self.angles_min = angle_min
        self.angles_max = angle_max
        self.grades_min = grades_min
        self.grades_max = grades_max
        self.latent_dim = latent_dim
        self.encoder = Encoder(H, W, latent_dim)
        self.decoder = Decoder(H, W, self.encoder, latent_dim)

    def sample_z(self, mu, logvar):
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    def forward(self, x, cond):
        mu, logvar = self.encoder(x, cond)
        z = self.sample_z(mu, logvar)
        logits = self.decoder(z, cond)  # (N,5,H,W)
        return logits, mu, logvar

def train(model, loader, epochs=20):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    print("Using {} device".format(device))
    model.to(device)

    # compute class weights once
    counts = Counter()
    for x_onehot, cond, labels in loader:
        counts.update(labels.view(-1).numpy().tolist())
    total = sum(counts.values())
    freqs = [counts[i] / total for i in range(5)]
    weights = torch.tensor([1.0 / (f + 1e-12) for f in freqs], dtype=torch.float32).to(device)
    weights = weights / weights.mean()

    weights =torch.tensor([1/proportion for proportion in loader.dataset.proportions], dtype=torch.float32).to(device)
    weights = weights / weights.mean()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    latent_dim = model.latent_dim
    kl_anneal_epochs = 20

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = epoch_recon = epoch_kl = 0.0
        kl_weight = min(1.0, epoch / kl_anneal_epochs)
        for x_onehot, cond, labels in loader:
            x_onehot = x_onehot.to(device)
            cond = cond.to(device)
            labels = labels.to(device).squeeze(1).long()

            logits, mu, logvar = model(x_onehot, cond)
            total_loss = loss_fn(logits, labels, mu, logvar, weight=weights, kl_weight=kl_weight)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()


        print(
            f"Epoch {epoch}  loss={epoch_loss / len(loader):.4f}")

        # debug sampling once per epoch
        with torch.no_grad():
            z = torch.randn(1, latent_dim, device=device)
            # use mean cond (or some test cond)
            cond_test = torch.tensor([[(model.angles_min + model.angles_max) / 2.0 - model.angles_min / (
                        model.angles_max - model.angles_min + 1e-8),
                                       (model.grades_min + model.grades_max) / 2.0 - model.grades_min / (
                                                   model.grades_max - model.grades_min + 1e-8)]]).to(device)
            logits = model.decoder(z, cond_test)
            probs = torch.softmax(logits, dim=1)
            print("sample probs mean per class:", probs.mean(dim=(0, 2, 3)).cpu().numpy())
            pred = logits.argmax(dim=1)
            print("unique pred sample:", torch.unique(pred))

def generate(model, angle, grade):

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    print("Using {} device".format(device))

    model.eval()
    # Normalize conditioning inputs
    a = (angle - model.angles_min) / (model.angles_max - model.angles_min + 1e-8)
    g = (grade - model.grades_min) / (model.grades_max - model.grades_min + 1e-8)
    cond = torch.tensor([[a, g]], dtype=torch.float32)

    # Sample latent vector
    z = torch.randn(1, model.latent_dim)

    z = z.to(device)
    cond = cond.to(device)

    with torch.no_grad():
        logits = model.decoder(z, cond)  # NOTE: decoder takes only ONE argument now

    # logits: (1, 5, H, W)
    classes = logits.argmax(dim=1)
    print("logits stats:", logits.min().item(), logits.max().item())
    print("unique argmax:", torch.unique(classes))

    return classes[0].cpu().numpy().astype(np.int8)








