import sys
from pathlib import Path

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, random_split

# Allow running this file directly (`python src/MainTrainGrader.py`, or an
# IDE's Run button) as well as as a module. See src/Main.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai.grader.GradeRegressor import GradeRegressor, GradeDataset, train_grader, evaluate_mae
from src.utils.Utils import save_grader_model

name = "16 x 12"
description = "Super Wide"

dataset = GradeDataset(name, description)

# Three-way split: train_set fits the weights, val_set is watched every epoch
# during training (monitoring/early-stopping signal), and test_set is held out
# untouched until the very end - the val_mae printed during training gets
# mildly optimistic the moment any decision is made by looking at it (even
# just "does this look converged"), so it's test_mae below, not val_mae, that
# should back "how much to trust this grader" when scoring generated climbs.
val_size = max(1, int(0.15 * len(dataset)))
test_size = max(1, int(0.15 * len(dataset)))
train_size = len(dataset) - val_size - test_size
train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

model = GradeRegressor(dataset.V, dataset.angles_min, dataset.angles_max, dataset.grades_min, dataset.grades_max)
history = train_grader(model, train_loader, val_loader=val_loader, epochs=30)

# Save right after training, before the plot - see src/MainTrain.py for why.
save_grader_model(model, name, description)

test_mae = evaluate_mae(model, test_loader)
print(f"\nHeld-out test MAE: {test_mae:.2f} grade points "
      f"(this is the number to trust, not the training-time val_mae above)")

epochs_seen = [h["epoch"] for h in history]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(epochs_seen, [h["mse"] for h in history])
ax1.set_xlabel("epoch")
ax1.set_ylabel("train MSE (normalized grade)")
if "val_mae" in history[0]:
    ax2.plot(epochs_seen, [h["val_mae"] for h in history])
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("validation MAE (grade points)")
fig.suptitle(f"Grader training - {name} {description}")
plt.tight_layout()
plt.show()
