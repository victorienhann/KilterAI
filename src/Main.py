"""Entry point to (re)build the training dataset for a board from the raw
climbs database. Run this whenever the source database changes or you want to
regenerate resources/datasets/<name>_<description>.h5. Training (MainTrain.py)
and generation (MainGenerate.py) read that file and don't need to re-run this.
"""
import sys
from pathlib import Path

# Allow running this file directly (`python src/Main.py`, or an IDE's Run
# button) as well as as a module (`python -m src.Main`). Running a file
# directly only puts its own directory (src/) on sys.path, not the repo root,
# which `from src...` imports need.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import Utils, Board

board_type = "kilter"
connection = Utils.connect_to_database(board_type)

name = "16 x 12"
description = "Super Wide"
board = Board.create_board(connection, board_type, name, description)

# For quick iteration/testing, cap how many climbs get used - set to None to
# use every available climb (this is what should run before real training).
limit = None
board.build_dataset(limit=limit)
