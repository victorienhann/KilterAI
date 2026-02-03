from src.utils import Utils, Board

board_type = "kilter"
connection = Utils.connect_to_database(board_type)
name = "16 x 12"
description = "Super Wide"
board = Board.create_board(connection, board_type, name, description)

board.build_dataset()

# TODO : model.train(dataset) (creation ,entrainement et sauvegarde)

# TODO : model.predict()
