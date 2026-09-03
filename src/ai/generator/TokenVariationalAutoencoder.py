"""Conditional VAE that generates climbs as picks over a board's fixed hold
vocabulary, instead of a dense pixel grid.

A board only has a few hundred physical hold positions (~692 for the 16x12
Super Wide kilter board), and a climb only ever uses a dozen or so of them.
Representing that as a 176x164 image made ~97% of the canvas structurally
meaningless (never a valid hold location) on top of the inherent sparsity of
climbs themselves. Here each climb is instead a length-V vector (V = the
board's vocab_size, see Board.create_token_vector), one entry per hold, valued
0 (unused) / 1 (start) / 2 (middle) / 3 (finish) / 4 (foot). That shrinks the
output space roughly 40x and removes the fake spatial sparsity, while keeping
the presence/type split (a hold is either used or not; if used, what role) that
proved useful for handling the remaining real imbalance (~13 active holds out
of ~692 slots).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm


class TokenDataset(Dataset):
    def __init__(self, name, description):
        from src.utils.Utils import load_dataset
        f = load_dataset(name, description)

        tokens = torch.tensor(f["tokens"][:], dtype=torch.long)  # (N, V), values 0..4
        self.V = tokens.shape[1]

        self.presence = (tokens > 0).long()  # (N, V)
        self.type = tokens.clone()  # (N, V), values 0..4 (0 where unused)

        presence_1hot = self.presence.unsqueeze(-1).float()  # (N, V, 1)
        type_1hot = F.one_hot(self.type, num_classes=5)[..., 1:5].float()  # (N, V, 4)
        self.x = torch.cat([presence_1hot, type_1hot], dim=-1)  # (N, V, 5): model input

        angles = torch.tensor(f["angles"][:], dtype=torch.float32)
        grades = torch.tensor(f["grades"][:], dtype=torch.float32)

        self.angles_min = angles.min()
        self.angles_max = angles.max()
        self.grades_min = grades.min()
        self.grades_max = grades.max()

        angles = (angles - self.angles_min) / (self.angles_max - self.angles_min + 1e-8)
        grades = (grades - self.grades_min) / (self.grades_max - self.grades_min + 1e-8)
        self.cond = torch.stack([angles, grades], dim=1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.cond[i], self.presence[i], self.type[i]


class Encoder(nn.Module):
    def __init__(self, vocab_size, latent_dim=16, hidden=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.net = nn.Sequential(
            nn.Linear(vocab_size * 5 + 2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)

    def forward(self, x, cond):
        h = x.reshape(x.size(0), -1)
        h = torch.cat([h, cond], dim=1)
        h = self.net(h)
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self, vocab_size, latent_dim=16, hidden=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.presence_head = nn.Linear(hidden, vocab_size)       # logits (N, V)
        self.type_head = nn.Linear(hidden, vocab_size * 4)       # logits (N, V, 4)

    def forward(self, z, cond):
        h = torch.cat([z, cond], dim=1)
        h = self.net(h)
        presence_logits = self.presence_head(h)
        type_logits = self.type_head(h).view(-1, self.vocab_size, 4)
        return presence_logits, type_logits


class TokenVariationalAutoEncoder(nn.Module):
    def __init__(self, vocab_size, angle_min, angle_max, grades_min, grades_max, latent_dim=16):
        super().__init__()
        self.vocab_size = vocab_size
        self.angles_min = angle_min
        self.angles_max = angle_max
        self.grades_min = grades_min
        self.grades_max = grades_max
        self.latent_dim = latent_dim
        self.encoder = Encoder(vocab_size, latent_dim)
        self.decoder = Decoder(vocab_size, latent_dim)

    def sample_z(self, mu, logvar):
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    def forward(self, x, cond):
        mu, logvar = self.encoder(x, cond)
        z = self.sample_z(mu, logvar)
        presence_logits, type_logits = self.decoder(z, cond)
        return presence_logits, type_logits, mu, logvar


def loss_fn(presence_logits, type_logits, presence_targets, type_targets, mu, logvar,
            pos_weight=None, type_weight=None, kl_weight=1.0):
    """
    presence_logits: (N,V) raw logits
    type_logits: (N,V,4) raw logits for classes 1..4
    presence_targets: (N,V) 0/1
    type_targets: (N,V) 0..4 (0 where no hold)
    """
    presence_loss = F.binary_cross_entropy_with_logits(
        presence_logits, presence_targets.float(), pos_weight=pos_weight)

    mask = presence_targets == 1
    if mask.sum() > 0:
        logits_sel = type_logits[mask]           # (M, 4)
        labels_sel = type_targets[mask] - 1       # (M,) 1..4 -> 0..3
        type_loss = F.cross_entropy(logits_sel, labels_sel, weight=type_weight)
    else:
        type_loss = torch.tensor(0.0, device=presence_logits.device)

    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = presence_loss + type_loss + kl_weight * kl
    return total, presence_loss.detach(), type_loss.detach(), kl.detach()


def _pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train(model, loader, epochs=30, lr=1e-3, device=None):
    """Trains the model in place. Returns the per-epoch loss history (a list of
    dicts: epoch, loss, presence, type, kl), e.g. for plotting loss vs epoch
    afterward with matplotlib."""
    device = device or _pick_device()
    print(f"Using {device} device")
    model.to(device)

    # Class weights from real counts over the whole dataset (presence pos_weight,
    # and inverse-frequency weights for the 4 hold roles).
    total_slots = 0
    total_presence = 0
    counts_type = torch.zeros(4)
    for _, _, presence, types in loader:
        total_slots += presence.numel()
        total_presence += presence.sum().item()
        present_types = types[presence == 1]
        for c in range(1, 5):
            counts_type[c - 1] += (present_types == c).sum().item()

    # Raw inverse-frequency weights are aggressive here (presence alone works out
    # to ~53x, since real holds are ~1.85% of slots), which pushes an undertrained
    # model to hedge by over-predicting broadly - worst on whichever classes get
    # the heaviest weight (finish is the rarest hold role, then start). Tempering
    # with sqrt (and a safety cap) tones that down; same technique the earlier
    # VariationalAutoEncoderBis prototype used for its presence pos_weight.
    pos_ratio = total_presence / total_slots
    raw_pos_weight = (1.0 - pos_ratio) / (pos_ratio + 1e-8)
    pos_weight = torch.tensor(min(raw_pos_weight ** 0.5, 20.0), dtype=torch.float32).to(device)

    type_freqs = (counts_type / counts_type.sum()).clamp(min=1e-8)
    type_weight = (1.0 / type_freqs) ** 0.5
    type_weight = (type_weight / type_weight.mean()).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    kl_anneal_epochs = max(1, epochs // 5)

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = epoch_presence = epoch_type = epoch_kl = 0.0
        kl_weight = min(1.0, epoch / kl_anneal_epochs)

        # A fresh bar per epoch, over that epoch's batches: shows this epoch's own
        # progress/ETA (from the batch rate) and leaves a completed line behind
        # once it's done, so all past epochs stay visible in the terminal.
        batch_bar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", unit="batch", leave=True)
        for i, (x, cond, presence, types) in enumerate(batch_bar, start=1):
            x, cond = x.to(device), cond.to(device)
            presence = presence.to(device).float()
            types = types.to(device).long()

            presence_logits, type_logits, mu, logvar = model(x, cond)
            loss, pres_l, type_l, kl = loss_fn(
                presence_logits, type_logits, presence, types, mu, logvar,
                pos_weight=pos_weight, type_weight=type_weight, kl_weight=kl_weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_presence += pres_l.item()
            epoch_type += type_l.item() if isinstance(type_l, torch.Tensor) else float(type_l)
            epoch_kl += kl.item()

            batch_bar.set_postfix(loss=f"{epoch_loss/i:.4f}", presence=f"{epoch_presence/i:.4f}",
                                   type=f"{epoch_type/i:.4f}", kl=f"{epoch_kl/i:.4f}", kl_w=f"{kl_weight:.2f}")

        n = len(loader)
        history.append({"epoch": epoch, "loss": epoch_loss / n, "presence": epoch_presence / n,
                         "type": epoch_type / n, "kl": epoch_kl / n})

    return history


def generate(model, angle, grade, threshold=0.5, device=None):
    """Returns a length-vocab_size numpy array with values 0..4 (see create_token_vector)."""
    device = device or next(model.parameters()).device
    model.eval()

    a = (angle - model.angles_min) / (model.angles_max - model.angles_min + 1e-8)
    g = (grade - model.grades_min) / (model.grades_max - model.grades_min + 1e-8)
    cond = torch.tensor([[a, g]], dtype=torch.float32).to(device)
    z = torch.randn(1, model.latent_dim, device=device)

    with torch.no_grad():
        presence_logits, type_logits = model.decoder(z, cond)
        presence = torch.sigmoid(presence_logits)[0] >= threshold  # (V,)
        types = torch.argmax(torch.softmax(type_logits, dim=-1), dim=-1)[0] + 1  # (V,) in 1..4

        result = torch.zeros(model.vocab_size, dtype=torch.long, device=device)
        result[presence] = types[presence]

    print("holds generated:", int(presence.sum().item()))
    return result.cpu().numpy().astype(np.int8)
