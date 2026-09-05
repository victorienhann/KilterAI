"""Generate several climbs at the same (angle, grade) and show them side by
side, to see how much shape variety the generator actually produces. Each
generate() call samples a fresh random latent z, so different draws should
give different hold layouts - this is the place to look if generated climbs
feel too similar to each other (e.g. if the model has collapsed to mostly
ignoring z and leaning on the angle/grade conditioning alone).
"""
import sys
from pathlib import Path

from matplotlib import pyplot as plt

# Allow running this file directly (`python src/MainVisualizeVariety.py`, or
# an IDE's Run button) as well as as a module. See src/Main.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai.generator.TokenVariationalAutoencoder import generate
from src.utils import Board, Utils
from src.utils.Utils import load_model
from src.utils.Grades import GRADES

board_type = "kilter"
connection = Utils.connect_to_database(board_type)

name = "16 x 12"
description = "Super Wide"
board = Board.create_board(connection, board_type, name, description)

model = load_model(name, description)

angle = 20
grade = "6b"
rows, cols = 2, 3

fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
for ax in axes.flat:
    matrix = generate(model, angle, GRADES[grade], verbose=False, threshold=0.4, second_pick_min_prob=0.5)
    n_holds = int((matrix > 0).sum())
    ax.imshow(board.visualize_climb(matrix))
    ax.set_title(f"{n_holds} holds")
    ax.axis("off")

fig.suptitle(f"{rows * cols} generations at angle={angle} grade={grade}")
plt.tight_layout()
plt.show()
