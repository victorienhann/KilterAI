import sys
from pathlib import Path

from matplotlib import pyplot as plt

# Allow running this file directly (`python src/MainGenerate.py`, or an IDE's
# Run button) as well as as a module (`python -m src.MainGenerate`). See src/Main.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import Utils, Board
from src.utils.Grades import GRADES
from src.utils.Utils import load_model, load_grader_model
from src.ai.generator.TokenVariationalAutoencoder import generate
from src.ai.grader.GradeRegressor import evaluate_climb

board_type = "kilter"
connection = Utils.connect_to_database(board_type)

name = "16 x 12"
description = "Super Wide"
board = Board.create_board(connection, board_type, name, description)


model = load_model(name, description)

angle = 20
grade = "6b"

matrix = generate(model, angle, GRADES[grade], threshold=0.3, second_pick_min_prob=0.3)

# Score the generated climb against the grade it was asked for, using the
# grader trained by MainTrainGrader.py - a way to sanity-check generation
# settings without eyeballing the render each time. Optional: skip cleanly if
# it hasn't been trained yet.
try:
    grader = load_grader_model(name, description)
    result = evaluate_climb(grader, matrix, angle, grade)
    print(f"Asked for {result['target_label']} (~{result['target_grade']}), "
          f"grader estimates {result['predicted_label']} (~{result['predicted_grade']:.1f}), "
          f"abs_error={result['abs_error']:.2f}")
except FileNotFoundError:
    print("No grader model found yet - run src/MainTrainGrader.py first to enable scoring.")

board.visualize_climb(matrix).show()