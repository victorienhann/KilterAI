"""Grade predictor: given a climb (as a hold-vocabulary token vector, see
Board.create_token_vector) and an angle, predicts its grade.

This exists to evaluate the generator rather than to be a product on its own:
TokenVariationalAutoencoder.generate() has several tunable parameters
(threshold, second_pick_min_prob, how long it was trained, ...) with no
ground truth to check them against - a generated climb doesn't have a "real"
grade to compare to. This model gives a proxy: train it on the same climbs
the generator learns from (so it learns the same "what makes a climb feel
like grade X" signal from data) and use its opinion on a *generated* climb as
a stand-in for "does this actually look like the grade it was asked for?".
See evaluate_climb() below.

Grade is treated as a regression target (denormalized back to the real grade
scale) rather than a classification target: grades are ordinal and the class
distribution is heavily skewed (some grades have only a handful of examples
in the training data), both of which regression handles better - and being
off by "one grade step" should count as a much smaller error than being off
by ten, which only a regression loss (not per-class cross-entropy) reflects.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm


class GradeDataset(Dataset):
    def __init__(self, name, description):
        from src.utils.Utils import load_dataset
        f = load_dataset(name, description)

        tokens = torch.tensor(f["tokens"][:], dtype=torch.long)  # (N, V), values 0..4
        self.V = tokens.shape[1]

        # Same (presence, type) one-hot encoding as TokenDataset, so a climb
        # means the same thing to both models.
        presence_1hot = (tokens > 0).unsqueeze(-1).float()  # (N, V, 1)
        type_1hot = F.one_hot(tokens, num_classes=5)[..., 1:5].float()  # (N, V, 4)
        self.x = torch.cat([presence_1hot, type_1hot], dim=-1)  # (N, V, 5): model input

        angles = torch.tensor(f["angles"][:], dtype=torch.float32)
        grades = torch.tensor(f["grades"][:], dtype=torch.float32)

        self.angles_min = angles.min()
        self.angles_max = angles.max()
        self.grades_min = grades.min()
        self.grades_max = grades.max()

        # Angle is an input feature here (unlike the generator, where it's a
        # conditioning input alongside grade) - grade is what we're predicting.
        self.angle = ((angles - self.angles_min) / (self.angles_max - self.angles_min + 1e-8)).unsqueeze(1)
        self.grade = (grades - self.grades_min) / (self.grades_max - self.grades_min + 1e-8)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.angle[i], self.grade[i]


class GradeRegressor(nn.Module):
    def __init__(self, vocab_size, angle_min, angle_max, grade_min, grade_max, hidden=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.angles_min = angle_min
        self.angles_max = angle_max
        self.grades_min = grade_min
        self.grades_max = grade_max
        self.net = nn.Sequential(
            nn.Linear(vocab_size * 5 + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x, angle):
        h = x.reshape(x.size(0), -1)
        h = torch.cat([h, angle], dim=1)
        return self.net(h).squeeze(-1)  # (N,), normalized grade


def _pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def evaluate_mae(model, loader, device=None):
    """Mean absolute grade error (real grade points, not normalized) of `model`
    over `loader`. Call this on a held-out test set the model never saw during
    training - not the same loader used as val_loader during train_grader -
    for a trustworthy final number: a validation set that gets watched every
    epoch during training (as val_loader is) stops being an unbiased estimate
    of generalization the moment any decision gets made by looking at it, even
    just "does this look converged"."""
    device = device or next(model.parameters()).device
    model.eval()
    grade_range = (model.grades_max - model.grades_min).item()
    abs_err = 0.0
    n_seen = 0
    with torch.no_grad():
        for x, angle, grade in loader:
            x, angle, grade = x.to(device), angle.to(device), grade.to(device)
            pred = model(x, angle)
            abs_err += (pred - grade).abs().sum().item() * grade_range  # denormalize
            n_seen += grade.numel()
    return abs_err / n_seen


def train_grader(model, train_loader, val_loader=None, epochs=30, lr=1e-3, device=None):
    """Trains the model in place. Returns the per-epoch history (list of dicts:
    epoch, mse, and val_mae in real grade units if val_loader is given).
    val_mae here is for watching training progress (e.g. spotting overfitting
    as it happens) - use evaluate_mae() on a separate held-out test set for
    the number that actually backs "how much to trust this grader" (see
    evaluate_climb)."""
    device = device or _pick_device()
    print(f"Using {device} device")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        batch_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch")
        for i, (x, angle, grade) in enumerate(batch_bar, start=1):
            x, angle, grade = x.to(device), angle.to(device), grade.to(device)

            pred = model(x, angle)
            loss = mse_loss(pred, grade)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_bar.set_postfix(mse=f"{epoch_loss/i:.5f}")

        entry = {"epoch": epoch, "mse": epoch_loss / len(train_loader)}

        if val_loader is not None:
            entry["val_mae"] = evaluate_mae(model, val_loader, device=device)
            print(f"  val_mae={entry['val_mae']:.2f} grade points")

        history.append(entry)

    return history


def predict_grade(model, tokens, angle, device=None):
    """tokens: array-like of length model.vocab_size, values 0..4 (a climb, see
    Board.create_token_vector). angle: raw angle in degrees. Returns the
    predicted grade on the real scale (e.g. ~18 for a 6b-ish climb)."""
    device = device or next(model.parameters()).device
    model.eval()

    tokens_t = torch.as_tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)  # (1, V)
    presence_1hot = (tokens_t > 0).unsqueeze(-1).float()
    type_1hot = F.one_hot(tokens_t, num_classes=5)[..., 1:5].float()
    x = torch.cat([presence_1hot, type_1hot], dim=-1)

    a = (angle - model.angles_min) / (model.angles_max - model.angles_min + 1e-8)
    angle_t = torch.tensor([[a]], dtype=torch.float32, device=device)

    with torch.no_grad():
        pred_norm = model(x, angle_t).item()

    return pred_norm * (model.grades_max - model.grades_min).item() + model.grades_min.item()


def evaluate_climb(grader_model, tokens, angle, target_grade, device=None):
    """Scores one generated climb against the grade it was generated for.
    target_grade may be a GRADES label (e.g. "6b") or a raw numeric grade.

    Use this to compare generation settings (threshold, second_pick_min_prob,
    how long the generator was trained, ...): generate a batch of climbs at a
    fixed (angle, grade), evaluate each, and look at the average abs_error -
    lower means the generator is more reliably producing climbs that read as
    the grade they were asked for, according to the grader's opinion (only as
    good as the grader's own val_mae from training - see train_grader)."""
    from src.utils.Grades import GRADES, grade_label

    target = GRADES[target_grade] if isinstance(target_grade, str) else target_grade
    predicted = predict_grade(grader_model, tokens, angle, device=device)
    return {
        "target_grade": target,
        "target_label": grade_label(target),
        "predicted_grade": predicted,
        "predicted_label": grade_label(predicted),
        "abs_error": abs(predicted - target),
    }
