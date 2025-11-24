from matplotlib import pyplot as plt

from src.utils import Utils, Board
from src.utils.Grades import GRADES
from src.utils.Utils import load_model
from src.ai.generator.VariationalAutoencoder import generate

board_type = "kilter"
connection = Utils.connect_to_database(board_type)

name = "16 x 12"
description = "Super Wide"
board = Board.create_board(connection, board_type, name, description)


model = load_model(name, description)

angle = 35
grade = "6b"

matrix = generate(model, angle, GRADES[grade])

board.visualize_climb(matrix).show()