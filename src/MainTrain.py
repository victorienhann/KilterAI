import sys
from pathlib import Path

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

# Allow running this file directly (`python src/MainTrain.py`, or an IDE's Run
# button) as well as as a module (`python -m src.MainTrain`). See src/Main.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai.generator.TokenVariationalAutoencoder import TokenVariationalAutoEncoder, TokenDataset, train, generate
from src.utils import Utils, Board
from src.utils.Grades import GRADES
from src.utils.Utils import save_model

board_type = "kilter"
name = "16 x 12"
description = "Super Wide"

# Only needed here to visualize the generated sample at the end - does NOT
# rebuild the dataset (that's src/Main.py's job, run separately and only when
# the source data changes).
connection = Utils.connect_to_database(board_type)
board = Board.create_board(connection, board_type, name, description)

dataset = TokenDataset(name, description)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

vocab_size = dataset.V
angle_min, angle_max = dataset.angles_min, dataset.angles_max
grades_min, grades_max = dataset.grades_min, dataset.grades_max
angle = 35
grade = "6b"

model = TokenVariationalAutoEncoder(vocab_size, angle_min, angle_max, grades_min, grades_max, latent_dim=16)
history = train(model, loader, epochs=10)

# Save right after training, before anything else that could fail (plotting,
# generating, visualizing) - a training run is expensive, a crash downstream
# shouldn't be able to lose it.
save_model(model, name, description)

matrix = generate(model, angle, GRADES[grade])

epochs_seen = [h["epoch"] for h in history]
plt.plot(epochs_seen, [h["loss"] for h in history], label="total")
plt.plot(epochs_seen, [h["presence"] for h in history], label="presence")
plt.plot(epochs_seen, [h["type"] for h in history], label="type")
plt.plot(epochs_seen, [h["kl"] for h in history], label="kl")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.title(f"Training loss - {name} {description}")
plt.show()

board.visualize_climb(matrix).show()
