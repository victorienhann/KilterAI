"""Grid-search generate()'s threshold and second_pick_min_prob against the
trained grader, to find settings that reliably produce climbs matching the
grade they were asked for. Needs both a trained generator (MainTrain.py) and
a trained grader (MainTrainGrader.py) to already exist.

For each (threshold, second_pick_min_prob) combination, generates several
climbs across a spread of (angle, grade) targets, scores each with the
grader, and reports the combination with the lowest mean absolute grade
error - averaging over multiple targets and draws rather than tuning against
a single point, since generation is stochastic (a fresh random z each call).
"""
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

# sys.path must be set up before any `from src...` import below can resolve
# when this file is run directly rather than as a module. See src/Main.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai.generator.TokenVariationalAutoencoder import generate
from src.ai.grader.GradeRegressor import evaluate_climb
from src.utils import Board, Utils
from src.utils.Utils import load_model, load_grader_model
from src.utils.Grades import GRADES

board_type = "kilter"
connection = Utils.connect_to_database(board_type)

name = "16 x 12"
description = "Super Wide"
board = Board.create_board(connection, board_type, name, description)

model = load_model(name, description)
grader = load_grader_model(name, description)

# A spread of representative (angle, grade) targets to average over, rather
# than overfitting the search to one specific point.
targets = [(angle, grade) for angle in (20,) for grade in ("6b",)]
samples_per_target = 3

threshold_grid = [i/100 for i in range(10, 90, 1)]
second_pick_grid = [i/100 for i in range(10, 90, 1)]


def mean_abs_error(threshold, second_pick_min_prob):
    errors = []
    for angle, grade in targets:
        for _ in range(samples_per_target):
            matrix = generate(model, angle, GRADES[grade], threshold=threshold,
                               second_pick_min_prob=second_pick_min_prob, verbose=False)
            errors.append(evaluate_climb(grader, matrix, angle, grade)["abs_error"])
    return float(np.mean(errors))


# Current default (threshold=None, adaptive top-k) as a baseline to compare
# the grid against - is a fixed threshold actually better than that?
baseline_error = mean_abs_error(None, 0.5)
print(f"baseline (adaptive threshold, second_pick_min_prob=0.5): mean_abs_error={baseline_error:.2f}\n")

results = np.zeros((len(threshold_grid), len(second_pick_grid)))
for ti, threshold in enumerate(threshold_grid):
    for si, second_pick_min_prob in enumerate(second_pick_grid):
        results[ti, si] = mean_abs_error(threshold, second_pick_min_prob)
        print(f"threshold={threshold:.1f}  second_pick_min_prob={second_pick_min_prob:.1f}  "
              f"mean_abs_error={results[ti, si]:.2f}")

best_ti, best_si = np.unravel_index(np.argmin(results), results.shape)
best_threshold = threshold_grid[best_ti]
best_second_pick = second_pick_grid[best_si]
best_error = results[best_ti, best_si]

print(f"\nBest grid setting: threshold={best_threshold:.1f} second_pick_min_prob={best_second_pick:.1f} "
      f"mean_abs_error={best_error:.2f}")
print(f"Baseline (adaptive):                                  mean_abs_error={baseline_error:.2f}")
if best_error < baseline_error:
    print("-> the grid setting beats the adaptive default; consider passing it explicitly to generate().")
else:
    print("-> the adaptive default (threshold=None) already does at least as well.")

fig, ax = plt.subplots()
im = ax.imshow(results, origin="lower", cmap="viridis_r",
                extent=[second_pick_grid[0], second_pick_grid[-1], threshold_grid[0], threshold_grid[-1]],
                aspect="auto")
ax.plot(best_second_pick, best_threshold, "r*", markersize=18, label="best")
ax.set_xlabel("second_pick_min_prob")
ax.set_ylabel("threshold")
ax.set_title(f"Mean abs grade error - {name} {description}")
ax.legend()
fig.colorbar(im, ax=ax, label="mean abs error (grade points)")
plt.tight_layout()
plt.show()

best_angle, best_grade = targets[0]
matrix = generate(model, best_angle, GRADES[best_grade], threshold=best_threshold, second_pick_min_prob=best_second_pick)

board.visualize_climb(matrix).show()
