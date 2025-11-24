import os
import time

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split

from src.database.KilterTorchDataset import collate_fn


# -----------------------------
# Training function
# -----------------------------
import time
from torch import optim
from torch.utils.data import DataLoader, random_split

def train_model(dataset, stats, model_class, num_epochs=10, batch_size=64, lr=1e-3, split_ratio=0.8, device="cpu"):
    train_size = int(len(dataset) * split_ratio)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = model_class(vocab_size=dataset.vocab_size).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    len_criterion = nn.CrossEntropyLoss()

    def normalize_dist_tensor(tensor, mean, std):
        return (tensor - mean) / (std + 1e-8)

    md_mean, md_std = stats["mean_dist_mean"], stats["mean_dist_std"]
    sd_mean, sd_std = stats["std_dist_mean"], stats["std_dist_std"]
    mx_mean, mx_std = stats["max_dist_mean"], stats["max_dist_std"]
    ttl_mean, ttl_std = stats["total_dist_mean"], stats["total_dist_std"]

    print("Training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        start_time = time.time()
        total_batches = len(train_loader)

        for i, batch in enumerate(train_loader):
            features_basic = batch["features"].to(device).float()
            mean_d = normalize_dist_tensor(batch["mean_dist"].to(device).float(), md_mean, md_std).unsqueeze(1)
            std_d = normalize_dist_tensor(batch["std_dist"].to(device).float(), sd_mean, sd_std).unsqueeze(1)
            max_d = normalize_dist_tensor(batch["max_dist"].to(device).float(), mx_mean, mx_std).unsqueeze(1)
            total_d = normalize_dist_tensor(batch["total_dist"].to(device).float(), ttl_mean, ttl_std).unsqueeze(1)

            inputs = torch.cat([features_basic, mean_d, std_d, max_d, total_d], dim=1)

            start_padded = batch["start"].to(device)
            middle_padded = batch["middle"].to(device)
            finish_padded = batch["finish"].to(device)
            foot_padded = batch["foot"].to(device)

            tgt_start = sequences_to_multihot_batch(start_padded, dataset.vocab_size, padding_idx=0)
            tgt_middle = sequences_to_multihot_batch(middle_padded, dataset.vocab_size, padding_idx=0)
            tgt_finish = sequences_to_multihot_batch(finish_padded, dataset.vocab_size, padding_idx=0)
            tgt_foot = sequences_to_multihot_batch(foot_padded, dataset.vocab_size, padding_idx=0)

            opt.zero_grad()
            out = model(inputs)

            # multi-hot binary loss
            loss = (
                criterion(out["start"], tgt_start)
                + criterion(out["middle"], tgt_middle)
                + criterion(out["finish"], tgt_finish)
                + criterion(out["foot"], tgt_foot)
            )

            # length prediction losses
            for key in ["start", "middle", "finish", "foot"]:
                logits = out[f"len_{key}"]
                targets = batch[f"len_{key}"].to(device).long() - 1  # recode 1..N -> 0..N-1
                targets = torch.clamp(targets, min=0, max=logits.shape[1] - 1)
                loss += len_criterion(logits, targets)

            loss.backward()
            opt.step()
            total_loss += loss.item()

            print_progress_bar(epoch, i, total_batches, total_loss, start_time)

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                features_basic = batch["features"].to(device).float()
                mean_d = normalize_dist_tensor(batch["mean_dist"].to(device).float(), md_mean, md_std).unsqueeze(1)
                std_d = normalize_dist_tensor(batch["std_dist"].to(device).float(), sd_mean, sd_std).unsqueeze(1)
                max_d = normalize_dist_tensor(batch["max_dist"].to(device).float(), mx_mean, mx_std).unsqueeze(1)
                total_d = normalize_dist_tensor(batch["total_dist"].to(device).float(), ttl_mean, ttl_std).unsqueeze(1)
                inputs = torch.cat([features_basic, mean_d, std_d, max_d, total_d], dim=1)

                out = model(inputs)
                start_padded = batch["start"].to(device)
                middle_padded = batch["middle"].to(device)
                finish_padded = batch["finish"].to(device)
                foot_padded = batch["foot"].to(device)

                tgt_start = sequences_to_multihot_batch(start_padded, dataset.vocab_size, padding_idx=0)
                tgt_middle = sequences_to_multihot_batch(middle_padded, dataset.vocab_size, padding_idx=0)
                tgt_finish = sequences_to_multihot_batch(finish_padded, dataset.vocab_size, padding_idx=0)
                tgt_foot = sequences_to_multihot_batch(foot_padded, dataset.vocab_size, padding_idx=0)

                val_loss += (
                    criterion(out["start"], tgt_start)
                    + criterion(out["middle"], tgt_middle)
                    + criterion(out["finish"], tgt_finish)
                    + criterion(out["foot"], tgt_foot)
                ).item()

        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} train_loss: {total_loss/len(train_loader):.6f} "
              f"val_loss: {val_loss/len(test_loader):.6f} elapsed: {elapsed:.1f}s")

    return model



import torch

def generate_sequences(model, features, stats, device="cpu", temperature=0.3):
    model.eval()
    with torch.no_grad():
        if features.dim() == 1:
            features = features.unsqueeze(0)

        def normalize_dist_tensor(tensor, mean, std):
            return (tensor - mean) / (std + 1e-8)

        md_mean, md_std = stats["mean_dist_mean"], stats["mean_dist_std"]
        sd_mean, sd_std = stats["std_dist_mean"], stats["std_dist_std"]
        mx_mean, mx_std = stats["max_dist_mean"], stats["max_dist_std"]
        ttl_mean, ttl_std = stats["total_dist_mean"], stats["total_dist_std"]

        mean_d = normalize_dist_tensor(torch.tensor([stats["mean_dist_mean"]]), md_mean, md_std).unsqueeze(0)
        std_d = normalize_dist_tensor(torch.tensor([stats["std_dist_mean"]]), sd_mean, sd_std).unsqueeze(0)
        max_d = normalize_dist_tensor(torch.tensor([stats["max_dist_mean"]]), mx_mean, mx_std).unsqueeze(0)
        total_d = normalize_dist_tensor(torch.tensor([stats["total_dist_mean"]]), ttl_mean, ttl_std).unsqueeze(0)

        inputs = torch.cat([features, mean_d, std_d, max_d, total_d], dim=1).to(device)
        out = model(inputs)

        # Tirage stochastique pour les longueurs
        def sample_length(logits):
            probs = torch.softmax(logits[0], dim=0)
            return torch.multinomial(probs, 1).item() + 1  # +1 car index 0 = 1 prise

        pred_len_start = sample_length(out["len_start"])
        pred_len_middle = sample_length(out["len_middle"])
        pred_len_finish = sample_length(out["len_finish"])
        pred_len_foot = sample_length(out["len_foot"])

        predictions = {}
        for key, max_len in zip(
            ["start", "middle", "finish", "foot"],
            [pred_len_start, pred_len_middle, pred_len_finish, pred_len_foot]
        ):
            logits = out[key][0] / temperature
            probs = torch.sigmoid(logits)
            probs = probs / probs.sum()

            # Tirage multinomial pondéré
            num_to_pick = min(max_len, len(probs))
            seq = torch.multinomial(probs, num_samples=num_to_pick, replacement=False).tolist()

            # Nettoyage
            seq = [s for s in seq if s != 0]  # exclure padding
            if len(seq) == 0 and key in ["start", "finish"]:
                seq = [torch.argmax(probs).item()]

            predictions[key] = seq

    return predictions

def save_model(model, path="resources/models/multihead_model.pth"):
    """
    Save trained model weights to a file.

    Args:
        model: trained MultiHeadGenerator
        path: destination file path
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"✅ Model weights saved to {path}")


def load_model(model_class, vocab_size, path="resources/models/multihead_model.pth", device="cpu", **kwargs):
    """
    Load model weights from file into a fresh model instance.

    Args:
        model_class: the model class (e.g., MultiHeadGenerator)
        vocab_size: vocabulary size used during models
        path: path to .pth file with saved weights
        device: "cpu" or "cuda"
        kwargs: other arguments needed to init the model (e.g. hidden_dim)

    Returns:
        model: loaded model ready for inference
    """
    model = model_class(vocab_size=vocab_size, **kwargs)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print(f"✅ Model weights loaded from {path}")
    return model

def compute_distance_stats(dataset):
    # dataset: instance de KilterTorchDataset
    import numpy as np
    means, stds, maxs, totals = [], [], [], []
    for i in range(len(dataset)):
        item = dataset[i]
        means.append(float(item["mean_dist"]))
        stds.append(float(item["std_dist"]))
        maxs.append(float(item["max_dist"]))
        totals.append(float(item["total_dist"]))
    stats = {
        "mean_dist_mean": float(np.mean(means)), "mean_dist_std": float(np.std(means)),
        "std_dist_mean": float(np.mean(stds)), "std_dist_std": float(np.std(stds)),
        "max_dist_mean": float(np.mean(maxs)), "max_dist_std": float(np.std(maxs)),
        "total_dist_mean": float(np.mean(totals)), "total_dist_std": float(np.std(totals)),
    }
    return stats

def format_eta(seconds):
    """Format seconds → h:mm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}"

def print_progress_bar(epoch, batch_idx, total_batches, avg_loss, start_time):
    """Affiche une barre de progression + ETA"""
    progress = (batch_idx + 1) / total_batches
    elapsed = time.time() - start_time
    eta = (elapsed / (batch_idx + 1)) * (total_batches - (batch_idx + 1))
    bar_length = 30
    filled = int(bar_length * progress)
    bar = "█" * filled + "-" * (bar_length - filled)
    print(
    f"\rEpoch {epoch+1} | [{bar}] {progress*100:5.1f}% | Loss {avg_loss:.4f} | ETA {format_eta(eta)}",
    end="",
    flush=True,
    )

import torch

def sequences_to_multihot_batch(padded_seqs: torch.LongTensor, vocab_size: int, padding_idx: int = 0) -> torch.FloatTensor:
    """
    padded_seqs: (batch, seq_len) long
    returns: (batch, vocab_size) float multi-hot
    Vectorized implementation using scatter_.
    """
    batch, seq_len = padded_seqs.shape
    device = padded_seqs.device

    # mask out padding
    mask = (padded_seqs != padding_idx)  # (batch, seq_len)
    # gather only valid indices (flattened)
    valid_idxs = padded_seqs * mask.long()  # zeros where padding
    # We'll create a (batch, vocab_size) zero tensor and scatter 1s at positions.
    multi = torch.zeros((batch, vocab_size), dtype=torch.float32, device=device)

    # For each batch element, unique indices are necessary to avoid duplicates,
    # but scatter_add with mask works: we'll set ones and then clamp to 1.
    # Create indices for scatter: rows repeated for each seq position
    rows = torch.arange(batch, device=device).unsqueeze(1).expand(-1, seq_len)  # (batch, seq_len)
    rows_flat = rows[mask].long()          # (N_valid,)
    cols_flat = valid_idxs[mask].long()    # (N_valid,)
    if cols_flat.numel() == 0:
        return multi
    multi.index_put_( (rows_flat, cols_flat), torch.ones_like(cols_flat, dtype=torch.float32), accumulate=True )
    # clamp to 1
    multi = torch.clamp(multi, 0.0, 1.0)
    return multi

import torch.nn as nn
import torch

class MultiHeadGenerator(nn.Module):
    def __init__(self, vocab_size, feature_size=6, embed_dim=128, hidden_dim=256, dropout=0.1):
        """
        Multi-label generator:
        - input features: angle, grade, mean_dist_norm, std_dist_norm, max_dist_norm, total_dist_norm
        - output heads: start/middle/finish/foot each (batch, vocab_size) logits
        """
        super().__init__()
        self.vocab_size = vocab_size

        self.input_fc = nn.Sequential(
            nn.Linear(feature_size, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU()
        )

        self.shared = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # heads
        self.start_out = nn.Linear(hidden_dim, vocab_size)
        self.middle_out = nn.Linear(hidden_dim, vocab_size)
        self.finish_out = nn.Linear(hidden_dim, vocab_size)
        self.foot_out = nn.Linear(hidden_dim, vocab_size)
        self.len_start_out = nn.Linear(hidden_dim, 2)  # pour prédire 1,2 → 3 classes
        self.len_middle_out = nn.Linear(hidden_dim, 11)  # 0 à 10
        self.len_finish_out = nn.Linear(hidden_dim, 2)
        self.len_foot_out = nn.Linear(hidden_dim, 11)

    def forward(self, features: torch.Tensor):
        """
        features: (batch, feature_size)
        returns dict logits (batch, vocab_size)
        """
        x = self.input_fc(features)
        h = self.shared(x)
        return {
            "start": self.start_out(h),
            "middle": self.middle_out(h),
            "finish": self.finish_out(h),
            "foot": self.foot_out(h),
            "len_start" : self.len_start_out(h),
            "len_middle" : self.len_middle_out(h),
            "len_finish" : self.len_finish_out(h),
            "len_foot" : self.len_foot_out(h),
            }


