import unittest

import pandas as pd

import src.utils.DataSetBuilder as dataset_builder
import src.utils.FramesParser as frames_parser
import src.visualizer.Visualizer as visualizer

from src.utils import Board
from src.visualizer import Visualizer


class VisualizerTest(unittest.TestCase):

    def test_get_images(self):
        db_path = "resources/databases/kilter.sqlite"
        builder = dataset_builder.DataSetBuilder(db_path)
        connection = builder.connect();
        query = """ SELECT image_filename, set_id FROM product_sizes_layouts_sets WHERE product_size_id = 28"""
        df = connection.execute(query)
        rows = df.fetchall()

        for row in rows:
            print(row)

    def test_visualize(self):
        db_path = "resources/databases/kilter.sqlite"
        builder = dataset_builder.DataSetBuilder(db_path)
        description = "Super Wide"
        name = "16 x 12"
        builder.extract_data(name, description)
        connection = builder.connect();
        query = """SELECT frames FROM climbs WHERE name = 'SAD' """
        df = pd.read_sql(query, connection)
        frames = df['frames']
        start, middle, finish, foot = frames_parser.parse_frames(frames)
        start_coords = [coord for coords in visualizer.get_xy_for_ids(builder.mapping, start).values() for coord in
                        coords]
        middle_coords = [coord for coords in visualizer.get_xy_for_ids(builder.mapping, middle).values() for coord in
                        coords]
        finish_coords = [coord for coords in visualizer.get_xy_for_ids(builder.mapping, finish).values() for coord in
                        coords]
        feet_coords = [coord for coords in visualizer.get_xy_for_ids(builder.mapping, foot).values() for coord in
                        coords]
        matrix = builder.create_matrix(start_coords, middle_coords, finish_coords, feet_coords)

        board = Board.create_board(connection, name, description)
        visu = visualizer.Visualizer(board, connection)
        visu.visualize_climb(matrix).show()

    def test_get_images(self):
        name = "Super Wide"
        db_path = "resources/databases/kilter.sqlite"
        builder = dataset_builder.DataSetBuilder(db_path)
        connection = builder.connect()
        edge_left = -24
        edge_right = 168
        edge_bottom = 0
        edge_top = 156
        visu = visualizer.Visualizer(name, connection)

    def test_get_edges(self):
        name = "Super Wide"
        db_path = "resources/databases/kilter.sqlite"
        builder = dataset_builder.DataSetBuilder(db_path)
        connection = builder.connect()
        edge_left = -24
        edge_right = 168
        edge_bottom = 0
        edge_top = 156
        l, r, b, t = visualizer.get_edges(name, connection)
        self.assertTrue(l == edge_left and r == edge_right and b == edge_bottom and t == edge_top)

    def test_get_images(self):
        board = "kilter"
        description = "Super Wide"
        name = "16 x 12"
        image_merge = Visualizer.get_template(board, name, description)
        image_merge.show()











