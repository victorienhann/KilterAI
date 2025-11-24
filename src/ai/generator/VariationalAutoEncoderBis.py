import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


# ---- MET À JOUR compute_proportions POUR PRESENCE ----
def compute_proportions_presence(matrices):
    total = matrices.numel()
    presence = (matrices > 0).sum().item() / total
    return presence


# ---- DATASET POUR ARCHITECTURE PRESENCE + TYPE ----
class TrainingDatasetBis(Dataset):
    def __init__(self, name, description):
        from src.utils.Utils import load_dataset
        f = load_dataset(name, description)

        matrices = torch.tensor(f["matrices"][:], dtype=torch.long)  # shape (N,H,W)

        self.H = matrices.shape[1]
        self.W = matrices.shape[2]

        # --- 1) Présence binaire ---
        # presence = 0 si element==0, sinon 1
        self.presence = (matrices > 0).long()  # (N,H,W)

        # --- 2) Type = classe 1..4, et 0 là où presence=0
        self.type = matrices.clone()  # (N,H,W)

        # --- 3) One-hot pour presence ---
        presence_1hot = self.presence.unsqueeze(1).float()  # (N,1,H,W)

        # --- 4) One-hot pour type : 4 classes seulement ---
        # classes: 1..4 → dim=5 → slice 1..4
        type_1hot_full = F.one_hot(self.type, num_classes=5)  # (N,H,W,5)
        type_1hot = type_1hot_full[..., 1:5].permute(0, 3, 1, 2).float()  # (N,4,H,W)

        # --- 5) INPUTS POUR LE VAE ---
        # On concatène les 1 canaux présence + 4 canaux type
        self.x = torch.cat([presence_1hot, type_1hot], dim=1)  # (N,5,H,W)

        # --- 6) TARGETS ---
        self.presence_targets = self.presence  # (N,H,W)
        self.type_targets = self.type          # (N,H,W)

        # --- ANGLES & GRADES ---
        angles = torch.tensor(f["angles"][:], dtype=torch.float32)
        grades = torch.tensor(f["grades"][:], dtype=torch.float32)

        self.angles_min, self.angles_max = angles.min(), angles.max()
        self.grades_min, self.grades_max = grades.min(), grades.max()

        angles = (angles - self.angles_min) / (self.angles_max - self.angles_min + 1e-8)
        grades = (grades - self.grades_min) / (self.grades_max - self.grades_min + 1e-8)

        self.cond = torch.stack([angles, grades], dim=1)  # (N,2)

        # ---- proportion de présence (utile si on veut contrer class imbalance) ----
        self.presence_proportion = compute_proportions_presence(matrices)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return (
            self.x[i],                       # entrées du modèle (N,5,H,W)
            self.cond[i],                    # (2,)
            self.presence_targets[i],        # (H,W)
            self.type_targets[i]             # (H,W)
        )

import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, H, W, latent_dim=16, in_channels=5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU()
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, H, W)
            out = self.conv(dummy)
            self.flat_dim = out.numel()

        self.fc = nn.Linear(self.flat_dim + 2, 256)
        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)

    def forward(self, x, cond):
        h = self.conv(x)                     # (N, C, Hc, Wc)
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
        self.encoder = encoder

        # compute compressed shape
        with torch.no_grad():
            dummy = torch.zeros(1, 5, H, W)
            out = self.encoder.conv(dummy)
            self.shape_after_conv = out.shape
            self.flat_dim = out.numel()

        self.fc = nn.Linear(latent_dim + 2, self.flat_dim)
        C, Hc, Wc = self.shape_after_conv[1:]
        self.C, self.Hc, self.Wc = C, Hc, Wc

        # shared deconv trunk
        self.shared_deconv = nn.Sequential(
            nn.ConvTranspose2d(C, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU()
        )

        # heads
        # head_presence -> 1 channel logits (binary)
        self.head_presence = nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1)
        # head_type -> 4 channel logits (classes 1..4)
        self.head_type = nn.ConvTranspose2d(16, 4, 4, stride=2, padding=1)

    def forward(self, z, cond):
        # z: (N, latent), cond: (N,2)
        h = torch.cat([z, cond], dim=1)
        h = F.relu(self.fc(h))
        h = h.view(-1, self.C, self.Hc, self.Wc)
        shared = self.shared_deconv(h)  # (N, 16, H', W')

        presence_logits = self.head_presence(shared)  # (N,1,H,W)
        type_logits = self.head_type(shared)          # (N,4,H,W)

        # ensure exact output size
        presence_logits = F.interpolate(presence_logits, size=(self.H, self.W), mode="bilinear", align_corners=False)
        type_logits = F.interpolate(type_logits, size=(self.H, self.W), mode="bilinear", align_corners=False)
        type_logits = type_logits / 0.5

        return presence_logits, type_logits


class VariationalAutoEncoderBis(nn.Module):
    def __init__(self, H, W, angle_min, angle_max, grades_min, grades_max, latent_dim=32):
        super().__init__()
        self.H = H; self.W = W
        self.angles_min = angle_min; self.angles_max = angle_max
        self.grades_min = grades_min; self.grades_max = grades_max
        self.latent_dim = latent_dim

        self.encoder = Encoder(H, W, latent_dim, in_channels=5)
        self.decoder = Decoder(H, W, self.encoder, latent_dim)

    def sample_z(self, mu, logvar):
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    def forward(self, x, cond):
        # x: (N,5,H,W) where channel0 presence, channels1-4 type one-hot
        mu, logvar = self.encoder(x, cond)
        z = self.sample_z(mu, logvar)
        presence_logits, type_logits = self.decoder(z, cond)
        return presence_logits, type_logits, mu, logvar

def loss_fn_presence_type(presence_logits, type_logits, presence_targets, type_targets,
                          mu, logvar,
                          bce_pos_weight=None,
                          type_weight=None,
                          kl_weight=1.0,
                          sparsity_lambda=0.001):
    """
    presence_logits: (N,1,H,W)  raw logits
    type_logits: (N,4,H,W) raw logits for classes 1..4
    presence_targets: (N,H,W)  0/1 long
    type_targets: (N,H,W)  0..4 long (0 where no hold)
    """
    device = presence_logits.device

    # 1) presence loss: BCE with logits
    if bce_pos_weight is not None:
        bce = nn.BCEWithLogitsLoss(pos_weight=bce_pos_weight)
        presence_loss = bce(presence_logits.squeeze(1), presence_targets.float())
    else:
        presence_loss = F.binary_cross_entropy_with_logits(presence_logits.squeeze(1), presence_targets.float())

    # 2) type loss: only where presence_targets==1
    mask = (presence_targets == 1)  # (N,H,W) bool
    if mask.sum() > 0:
        # select logits at masked positions
        # reshape to (N,4,H,W) -> (N*H*W,4)
        logits_flat = type_logits.permute(0,2,3,1).reshape(-1, 4)  # (N*H*W,4)
        labels_flat = type_targets.reshape(-1)  # (N*H*W,)
        mask_flat = mask.reshape(-1)

        logits_sel = logits_flat[mask_flat]
        labels_sel = labels_flat[mask_flat] - 1  # convert 1..4 -> 0..3 for CE
        # if type_weight provided, use it
        if type_weight is not None:
            ce = nn.CrossEntropyLoss(weight=type_weight.to(device))
            type_loss = ce(logits_sel, labels_sel)
        else:
            type_loss = F.cross_entropy(logits_sel, labels_sel)
    else:
        type_loss = torch.tensor(0.0, device=device)

    # 3) sparsity on presence probability (encourage low density)
    p = torch.sigmoid(presence_logits)  # (N,1,H,W)
    sparsity = p.mean()

    # 4) KL
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    total = presence_loss + 2 * type_loss + sparsity_lambda * sparsity + kl_weight * kl
    return total, presence_loss.detach(), type_loss.detach(), sparsity.detach(), kl.detach()

def trainBis(model, loader, epochs=50, lr=5e-5, device=None):
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print("Using", device)
    model.to(device)

    # compute bce pos_weight (for presence) and type class weights
    # pos_weight = (neg/pos) but tempered/clipped
    total_pixels = 0
    total_presence = 0
    counts_type = torch.zeros(4, dtype=torch.long)  # counts for classes 1..4
    for x, cond, presence_t, type_t in loader:
        total_pixels += presence_t.numel()
        total_presence += int((presence_t == 1).sum().item())
        # count types among present positions
        types = type_t[presence_t==1]
        if types.numel() > 0:
            for c in range(1,5):
                counts_type[c-1] += int((types==c).sum().item())

    # compute pos_weight (presence) more strongly but tempered
    eps = 1e-9
    pos_ratio = total_presence / (total_pixels + eps)
    neg_ratio = 1.0 - pos_ratio
    raw_pos_weight = (neg_ratio + eps) / (pos_ratio + eps)  # e.g. ~2000

    # Temper but less aggressively than cbrt; use sqrt for stronger effect
    pos_weight_val = float(raw_pos_weight ** 0.5)  # sqrt
    # Clip to avoid insane values
    pos_weight_val = max(1.0, min(pos_weight_val, 2000.0))
    bce_pos_weight = torch.tensor(pos_weight_val, dtype=torch.float32).to(device)
    print("presence pos_weight", pos_weight_val, "pos_ratio", pos_ratio)

    # type weights: inverse frequency but tempered
    counts_type = counts_type.float()
    type_freqs = (counts_type / counts_type.sum()).clamp(min=1e-9)
    type_raw = 1.0 / type_freqs
    type_weight = (type_raw / type_raw.mean()).to(device)  # normalized, tempered a bit
    print("type freqs:", type_freqs.cpu().numpy(), "type weights:", type_weight.cpu().numpy())

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    kl_anneal_epochs = max(1, epochs // 5)
    latent_dim = model.latent_dim

    for epoch in range(1, epochs+1):
        model.train()
        epoch_loss = 0.0
        epoch_presence_loss = 0.0
        epoch_type_loss = 0.0
        epoch_sparsity = 0.0
        epoch_kl = 0.0
        kl_weight = min(1.0, epoch / kl_anneal_epochs)

        for x, cond, presence_t, type_t in loader:
            x = x.to(device)
            cond = cond.to(device)
            presence_t = presence_t.to(device).long()
            type_t = type_t.to(device).long()

            presence_logits, type_logits, mu, logvar = model(x, cond)

            loss, pres_l, type_l, spars, kl = loss_fn_presence_type(
                presence_logits, type_logits, presence_t, type_t,
                mu, logvar,
                bce_pos_weight=bce_pos_weight,
                type_weight=type_weight,
                kl_weight=kl_weight,
                sparsity_lambda=0.5  # tuneable
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_presence_loss += pres_l.item()
            epoch_type_loss += type_l.item() if isinstance(type_l, torch.Tensor) else float(type_l)
            epoch_sparsity += spars.item()
            epoch_kl += kl.item()

        n_batches = len(loader)
        print(f"Epoch {epoch} loss={epoch_loss/n_batches:.4f} pres={epoch_presence_loss/n_batches:.4f} "
              f"type={epoch_type_loss/n_batches:.4f} sparsity={epoch_sparsity/n_batches:.4f} kl={epoch_kl/n_batches:.4f} kl_w={kl_weight:.3f}")

        # sample debug (one sample)
        with torch.no_grad():
            z = torch.randn(1, latent_dim, device=device)
            cond_test = torch.tensor([[(model.angles_min + model.angles_max)/2.0 - model.angles_min / (model.angles_max-model.angles_min+1e-8),
                                       (model.grades_min + model.grades_max)/2.0 - model.grades_min / (model.grades_max-model.grades_min+1e-8)]], dtype=torch.float32).to(device)
            p_logits, t_logits = model.decoder(z, cond_test)
            probs_presence = torch.sigmoid(p_logits)
            probs_type = torch.softmax(t_logits, dim=1)
            print("sample presence mean:", probs_presence.mean().item(), "type mean per class:", probs_type.mean(dim=(0,2,3)).cpu().numpy())

def generateBis(model, angle, grade, threshold=0.5, sample_presence=False):
    # returns numpy array (H,W) with values 0..4
    device = next(model.parameters()).device
    model.eval()

    a = (angle - model.angles_min) / (model.angles_max - model.angles_min + 1e-8)
    g = (grade - model.grades_min) / (model.grades_max - model.grades_min + 1e-8)
    cond = torch.tensor([[a, g]], dtype=torch.float32).to(device)

    z = torch.randn(1, model.latent_dim).to(device)

    with torch.no_grad():
        p_logits, t_logits = model.decoder(z, cond)  # (1,1,H,W), (1,4,H,W)
        p_probs = torch.sigmoid(p_logits).squeeze(0).squeeze(0)   # (H,W)
        t_probs = torch.softmax(t_logits, dim=1).squeeze(0)      # (4,H,W)

        if sample_presence:
            # bernoulli sample for stochasticity
            presence = torch.bernoulli(p_probs)
        else:
            presence = (p_probs >= threshold).float()

        types_argmax = torch.argmax(t_probs, dim=0).long()  # 0..3 mapping to original classes 1..4

        result = torch.zeros_like(types_argmax, dtype=torch.int64)
        # set result = type+1 where presence==1
        result[presence==1] = types_argmax[presence==1] + 1

    return result.cpu().numpy().astype(np.int8)

