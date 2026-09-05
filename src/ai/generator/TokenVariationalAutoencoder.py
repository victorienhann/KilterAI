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
            pos_weight=None, type_weight=None, kl_weight=1.0, free_bits=0.1):
    """
    presence_logits: (N,V) raw logits
    type_logits: (N,V,4) raw logits for classes 1..4
    presence_targets: (N,V) 0/1
    type_targets: (N,V) 0..4 (0 where no hold)

    free_bits: KL "budget" (in nats) per latent dimension that isn't
    penalized. Without this, the cheapest way to minimize the raw KL term is
    to collapse every dimension's posterior exactly onto the prior (mu=0,
    logvar=0) - z then carries no information about the specific input, the
    decoder learns to ignore it and reconstructs only from the (angle, grade)
    conditioning, and every sample at a given (angle, grade) comes out
    looking like the same "average" climb regardless of z (posterior
    collapse - a well-known VAE failure mode). Clamping each dimension's KL
    to at least free_bits removes the incentive to collapse below that
    budget, so the gradient pushing z toward being informative doesn't
    vanish to zero.
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

    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (N, latent_dim)
    kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    kl = kl_per_dim.sum(dim=1).mean()

    total = presence_loss + type_loss + kl_weight * kl
    return total, presence_loss.detach(), type_loss.detach(), kl.detach()


def _pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train(model, loader, epochs=30, lr=1e-3, device=None, free_bits=0.1):
    """Trains the model in place. Returns the per-epoch loss history (a list of
    dicts: epoch, loss, presence, type, kl), e.g. for plotting loss vs epoch
    afterward with matplotlib.

    free_bits: see loss_fn - raise it if generated samples still look too
    similar to each other after retraining (posterior collapse), lower it
    (toward 0) if training seems to spend too much capacity on KL at the
    expense of reconstruction quality (presence/type loss staying high)."""
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
    # Ramp over the first half of training rather than the first fifth - a
    # fast ramp reaches full KL pressure before the decoder has had much
    # chance to learn to lean on z, which pushes toward posterior collapse
    # (see loss_fn's free_bits docstring).
    kl_anneal_epochs = max(1, epochs // 2)

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
                pos_weight=pos_weight, type_weight=type_weight, kl_weight=kl_weight, free_bits=free_bits)

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


def _pick_role(candidates, scores, count_range, second_pick_min_prob):
    """Among `candidates` (hold indices), choose which get one role, ranked by
    `scores` (that role's probability for each candidate). Always takes the
    single best; takes a second only if count_range allows it and the model is
    at least second_pick_min_prob confident in it. Returns (chosen, remaining)
    - remaining being the candidates not chosen, for the next role to draw from."""
    if len(candidates) == 0:
        return candidates, candidates
    order = torch.argsort(scores[candidates], descending=True)
    ranked = candidates[order]
    n = 1
    if len(ranked) > 1 and count_range[1] > 1 and scores[ranked[1]] >= second_pick_min_prob:
        n = 2
    n = min(n, count_range[1], len(ranked))
    return ranked[:n], ranked[n:]


def generate(model, angle, grade, threshold=None, device=None,
             start_count_range=(1, 2), finish_count_range=(1, 2), second_pick_min_prob=0.5, verbose=True):
    """Returns a length-vocab_size numpy array with values 0..4 (see create_token_vector).

    Presence (which holds are used at all): by default (threshold=None), picks
    the k most likely holds, where k is the model's own expected count - the
    sum of its per-hold presence probabilities - rather than a fixed
    presence-probability cutoff. A fixed threshold is brittle: how many holds
    clear it depends on the absolute scale the presence loss happens to push
    probabilities to (e.g. how strongly pos_weight is tempered, how many
    epochs it's had), so it tends to swing between "too many holds" and "too
    few holds" as that scale shifts, rather than tracking what the model
    actually believes. Summing probabilities sidesteps that: it stays a
    reasonable count estimate even when the individual probabilities are
    shifted up or down together. Pass an explicit threshold (e.g. 0.5) to use
    the old fixed-cutoff behavior instead - role counts are still enforced
    either way.

    Roles (start/middle/finish/foot): every real climb in the training data
    has 1 or 2 start holds and 1 or 2 finish holds (Board.build_dataset only
    keeps climbs with len(starts)<=2 and len(finishes)<=2, and a climb always
    needs at least one of each to be climbable) - independent per-hold argmax
    has no way to guarantee that, so start/finish are instead each assigned
    their most-confident candidate(s) among the selected holds first (see
    start_count_range/finish_count_range/second_pick_min_prob), and only the
    leftover holds get a plain middle-vs-foot argmax.
    """
    device = device or next(model.parameters()).device
    model.eval()

    a = (angle - model.angles_min) / (model.angles_max - model.angles_min + 1e-8)
    g = (grade - model.grades_min) / (model.grades_max - model.grades_min + 1e-8)
    cond = torch.tensor([[a, g]], dtype=torch.float32).to(device)
    z = torch.randn(1, model.latent_dim, device=device)

    with torch.no_grad():
        presence_logits, type_logits = model.decoder(z, cond)
        presence_probs = torch.sigmoid(presence_logits)[0]  # (V,)
        type_probs = torch.softmax(type_logits, dim=-1)[0]  # (V,4): start,middle,finish,foot

        if threshold is not None:
            presence = presence_probs >= threshold
        else:
            k = round(presence_probs.sum().item())
            presence = torch.zeros_like(presence_probs, dtype=torch.bool)
            top_indices = torch.topk(presence_probs, min(model.vocab_size, max(k, 1))).indices
            presence[top_indices] = True

        # Need room for at least one start and one finish hold - top up with
        # the next most-likely holds if too few cleared presence selection.
        min_needed = start_count_range[0] + finish_count_range[0]
        if presence.sum().item() < min_needed:
            scored = presence_probs.masked_fill(presence, -1.0)
            extra = torch.topk(scored, min_needed - int(presence.sum().item())).indices
            presence[extra] = True

        pool = presence.nonzero(as_tuple=True)[0]
        result = torch.zeros(model.vocab_size, dtype=torch.long, device=device)

        start_idx, pool = _pick_role(pool, type_probs[:, 0], start_count_range, second_pick_min_prob)
        finish_idx, pool = _pick_role(pool, type_probs[:, 2], finish_count_range, second_pick_min_prob)
        result[start_idx] = 1
        result[finish_idx] = 3

        if len(pool) > 0:
            is_foot = type_probs[pool, 3] > type_probs[pool, 1]  # middle vs foot only
            result[pool] = torch.where(is_foot, torch.tensor(4, device=device), torch.tensor(2, device=device))

    if verbose:
        n_holds = int(presence.sum().item())
        print(f"holds generated: {n_holds} (start={len(start_idx)}, finish={len(finish_idx)}, "
              f"model's expected count: {presence_probs.sum().item():.1f})")
    return result.cpu().numpy().astype(np.int8)
