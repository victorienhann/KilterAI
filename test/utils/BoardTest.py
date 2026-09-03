import sqlite3
import unittest
import src.utils.Board as Board

class BoardTest(unittest.TestCase):

    def test_create_board(self):
        db_path = "resources/databases/kilter.sqlite"
        conn = sqlite3.connect(db_path)
        board_type = "kilter"
        description = "Super Wide"
        name = "16 x 12"
        board = Board.create_board(conn, board_type, name, description)
        layout_id = 1
        set_id = [1, 20]
        edge_left, edge_right, edge_bottom, edge_top = -24, 168, 0, 156
        self.assertEqual(board.name, name)
        self.assertEqual(board.description, description)
        self.assertEqual(board.layout_id, layout_id)
        self.assertEqual(board.set_id, set_id)
        self.assertEqual(board.edge_left, edge_left)
        self.assertEqual(board.edge_right, edge_right)
        self.assertEqual(board.edge_bottom, edge_bottom)
        self.assertEqual(board.edge_top, edge_top)

    def test_load_holds(self):
        db_path = "resources/databases/kilter.sqlite"
        conn = sqlite3.connect(db_path)
        board_type = "kilter"
        description = "Super Wide"
        name = "16 x 12"
        board = Board.create_board(conn, board_type, name, description)
        board.load_holds()
        self.assertEqual(board.vocab_size, 692)
        self.assertEqual(len(board.hold_x), board.vocab_size)
        self.assertEqual(len(board.hold_y), board.vocab_size)
