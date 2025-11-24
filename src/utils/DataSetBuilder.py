import os
import sqlite3

import numpy as np
import pandas as pd
import re

from tqdm import tqdm

from src.database.KilterDataSet import KilterDataset
from src.utils import Board
from src.utils.Queries import QUERIES
from src.utils.Roles import ROLES as roles
from src.visualizer.Visualizer import get_xy_for_ids




class DataSetBuilder:
    def __init__(self, path):
        self.db_path = path
        self.connection = None
        self.board = None
        self.mapping = None

    def connect(self):
        """Connect to SQLite database, raising error if file does not exist"""
        if not os.path.isfile(self.db_path):
            raise FileNotFoundError(f"Database file does not exist: {self.db_path}")
        try:
            self.connection = sqlite3.connect(self.db_path)
            print(f"Connected to database: {self.db_path}")
            return self.connection
        except sqlite3.Error as e:
            raise RuntimeError(f"Error connecting to database: {e}")
        
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            print("Database connection closed")

    def extract_data(self, name, description):
        """Extract climbs, angle and display_difficulty from the database"""
        if self.connection == None:
            self.connect()
        self.board = Board.create_board(self.connection, name, description)
        try:
            df = pd.read_sql(QUERIES["board_climb"], self.connect(), None, None, {"layout_id": self.board.get_layout_id(),
                             "edge_left" : self.board.get_edge_left(), "edge_right" : self.board.get_edge_right(),
                             "edge_bottom" : self.board.get_edge_bottom(), "edge_top" : self.board.get_edge_top()})
            print(f"Extracted {len(df)} rows from database")
            print("Mapping ...")
            self.mapping = pd.read_sql(QUERIES["holds"], self.connection, None, None, {'set_id_1' : self.board.get_set_id()[0], 'set_id_2' : self.board.get_set_id()[1], 'layout_id' : self.board.get_layout_id()})
            print("Mapping done")
            return df
        except Exception as e:
            print(f"Error executing query: {e}")
            return None

    def create_matrix(self, start, middle, finish, foot):
        x_max = np.max(self.mapping["x"])
        y_max = np.max(self.mapping["y"])
        matrix = np.zeros((y_max, x_max), dtype=np.float16)
        # We divide by 4 every coeficient to be in [0,1]
        for (x,y) in start:
            matrix[x,y] = 1/4
        for (x,y) in middle:
            matrix[x,y] = 2/4
        for (x,y) in finish:
            matrix[x,y] = 3/4
        for (x,y) in foot:
            matrix[x,y] = 4/4
        return matrix

    def build_dataset(self, name, description):
        df = self.extract_data(name, description)
        climbs = df['frames']
        matrices = []
        angles = []
        grades = []


        for i, climb in enumerate(tqdm(climbs, desc="Building dataset for board " + self.board.description + " " + self.board.get_name() , unit="rows")):
            #For test purpose
            if i > 1000 :
                break

            starts_climb, middles_climb, finishes_climb, feet_climb = extract_roles(climb)
            if len(starts_climb) <= 2 and len(finishes_climb) <= 2:
                start_coords = [coord for coords in get_xy_for_ids(self.mapping, starts_climb).values() for coord in
                                coords]
                middle_coords = [coord for coords in get_xy_for_ids(self.mapping, middles_climb).values() for coord in
                                 coords]
                finish_coords = [coord for coords in get_xy_for_ids(self.mapping, finishes_climb).values() for coord in
                                 coords]
                feet_coords = [coord for coords in get_xy_for_ids(self.mapping, feet_climb).values() for coord in
                               coords]
                matrix = self.create_matrix(start_coords, middle_coords, finish_coords, feet_coords)
                matrices.append(matrix)
                angles.append(df['angle'][i])
                grades.append(round(df['display_difficulty'][i]))


        dataset = KilterDataset(matrices, angles, grades)
        return dataset
