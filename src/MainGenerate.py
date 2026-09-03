import sys
from pathlib import Path

from matplotlib import pyplot as plt

# Allow running this file directly (`python src/MainGenerate.py`, or an IDE's
# Run button) as well as as a module (`python -m src.MainGenerate`). See src/Main.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import Utils, Board
from src.utils.Grades import GRADES
from src.utils.Utils import load_model
from src.ai.generator.TokenVariationalAutoencoder import generate

board_type = "kilter"
connection = Utils.connect_to_database(board_type)

name = "16 x 12"
description = "Super Wide"
board = Board.create_board(connection, board_type, name, description)


model = load_model(name, description)

angle = 20
grade = "6b"

matrix = generate(model, angle, GRADES[grade])

board.visualize_climb(matrix).show()