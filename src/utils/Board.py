import numpy as np
import pandas as pd
from tqdm import tqdm


from src.database.KilterDataSet import KilterDataset
from src.utils.Colors import COLORS
from src.utils.Queries import QUERIES
from src.utils.Utils import extract_roles, make_circle, save_dataset
from src.visualizer import Visualizer
from src.visualizer.Visualizer import get_xy_for_ids


def create_board(connection, board_type, name, description):
    df = pd.read_sql(QUERIES["board_info"], connection, None, None,{'name': name, 'description': description})
    layout_id = int(df['layout_id'].values[0])
    set_id = [int(df['set_id'].values[i]) for i in range(len(df['set_id'].values))]
    edge_left = int(df['edge_left'].values[0])
    edge_right = int(df['edge_right'].values[0])
    edge_bottom = int(df['edge_bottom'].values[0])
    edge_top = int(df['edge_top'].values[0])
    template = Visualizer.get_template(board_type, name, description)
    return Board(connection, name, description, layout_id, set_id, edge_left, edge_right, edge_bottom, edge_top, template)

class Board:
    def __init__(self, connection, name, description, layout_id, set_id, edge_left, edge_right, edge_bottom, edge_top, template):
        self.connection = connection
        self.name = name
        self.description = description
        self.layout_id = layout_id
        self.set_id = set_id
        self.edge_left = edge_left
        self.edge_right = edge_right
        self.edge_top = edge_top
        self.edge_bottom = edge_bottom
        self.template = template
        self.mapping = None

    def get_edges(self):
        return self.edge_left, self.edge_right, self.edge_bottom, self.edge_top

    def extract_data(self):
        """Extract climbs, angle and display_difficulty from the database"""
        try:
            df = pd.read_sql(QUERIES["board_climb"], self.connection, None, None, {"layout_id": self.layout_id,
                             "edge_left" : self.edge_left, "edge_right" : self.edge_right,
                             "edge_bottom" : self.edge_bottom, "edge_top" : self.edge_top})
            print(f"Found {len(df)} climbs for board {self.name} {self.description}")
            print("Mapping ...")
            self.mapping = pd.read_sql(QUERIES["holds"], self.connection, None, None,
                                       {'set_id_1' : self.set_id[0], 'set_id_2' : self.set_id[1], 'layout_id' : self.layout_id})
            print("Mapping done")
            return df
        except Exception as e:
            print(f"Error executing query: {e}")
            return None

    def create_matrix(self, start, middle, finish, foot):
        x_max = np.max(self.mapping["x"])
        y_max = np.max(self.mapping["y"])
        matrix = np.zeros((y_max, x_max), dtype=np.int8)
        # We divide by 4 every coeficient to be in [0,1]
        for (x,y) in start:
            matrix[x,y] = 1
        for (x,y) in middle:
            matrix[x,y] = 2
        for (x,y) in finish:
            matrix[x,y] = 3
        for (x,y) in foot:
            matrix[x,y] = 4
        return matrix

    def build_dataset(self):
        df = self.extract_data()
        climbs = df['frames']
        matrices = []
        angles = []
        grades = []


        for i, climb in enumerate(tqdm(climbs, desc=f"Building dataset for board {self.description} {self.name}" , unit="climbs")):
            #For test purpose
            #if i > 1000 :
                #break

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
        save_dataset(dataset, self.name, self.description)

    def visualize_climb(self, matrix):

        start_color = COLORS["start"]
        middle_color = COLORS["middle"]
        finish_color = COLORS["finish"]
        foot_color = COLORS["foot"]

        edge_left, edge_right, edge_bottom, edge_top = self.get_edges()
        # Getting the board image without markers
        image = self.template
        width, height = self.template.size

        xSpacing = width / (edge_right - edge_left)
        ySpacing = height / (edge_top - edge_bottom)

        for i in range(len(matrix)): # i is the Y coordinate
            for j in range(len(matrix[i])): # j is the X coordinate
                if matrix[i][j] > 0 :
                    xPixel = int((i - edge_left) * xSpacing)
                    yPixel = int(height - (j - edge_bottom) * ySpacing)
                    if matrix[i][j] == 1: # start hold
                        color = start_color
                    if matrix[i][j] == 2:
                        color = middle_color
                    if matrix[i][j] == 3:
                        color = finish_color
                    if matrix[i][j] == 4:
                        color = foot_color
                    circle = make_circle(color)
                    image.paste(circle, (xPixel - circle.width // 2, yPixel- circle.height // 2), circle)
        return image


