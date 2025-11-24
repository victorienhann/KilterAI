import numpy as np
import pandas as pd
from PIL import Image

from src.utils.Queries import QUERIES
from src.visualizer.Visualizer import get_xy_for_ids


class Climb:
    def __init__(self, start, middle, finish, foot):
        self.start = start
        self.middle = middle
        self.finish = finish
        self.foot = foot
        self.edge_left = None
        self.edge_right = None
        self.edge_top = None
        self.edge_bottom = None

    def __init__(self, start, middle, finish, foot, edge_left, edge_right, edge_top, edge_bottom):
        self.start = start
        self.middle = middle
        self.finish = finish
        self.foot = foot
        self.edge_left = edge_left
        self.edge_right = edge_right
        self.edge_top = edge_top
        self.edge_bottom = edge_bottom


    def get_start(self):
        return self.start

    def get_middle(self):
        return self.middle

    def get_finish(self):
        return self.finish

    def get_foot(self):
        return self.foot

    def get_edge_left(self):
        return self.edge_left

    def get_edge_right(self):
        return self.edge_right

    def get_edge_top(self):
        return self.edge_top

    def get_edge_bottom(self):
        return self.edge_bottom

    def get_holds_ids(self):
        return [i for i in self.get_start() + self.get_middle() + self.get_finish() + self.get_foot()]

    def get_distances(self, board, connection):
        holds = self.get_holds_ids()
        layout_id = board.get_layout_id()
        set_id = board.get_set_id()
        df = pd.read_sql(QUERIES["holds"], connection, None, None, {'set_id_1' : set_id[0], 'set_id_2' : set_id[1], 'layout_id' : layout_id})
        xy = get_xy_for_ids(df, holds)
        edge_left = board.get_edge_left()
        edge_right = board.get_edge_right()
        edge_top = board.get_edge_top()
        edge_bottom = board.get_edge_bottom()
        im_dir = "../resources/images/"
        images = board.get_images()
        im1_path = im_dir + images[0]
        im1 = Image.open(im1_path).convert('RGBA')
        width, height = im1.size

        xSpacing = width / (edge_right - edge_left)
        ySpacing = height / (edge_top - edge_bottom)

        x = [xy[id][0][0] for id in holds]
        y = [xy[id][0][1] for id in holds]
        coords = []
        for i in range(len(x)):
            xPixel = int((x[i] - edge_left) * xSpacing)
            yPixel = int(height - (y[i] - edge_bottom) * ySpacing)
            coords.append((xPixel, yPixel))

        metrics = {}
        metrics["n_points"] = len(coords)
        # Distance
        deltas = np.diff(coords, axis=0)
        dists = np.linalg.norm(deltas, axis=1)
        metrics["mean_dist"] = np.mean(dists)
        metrics["std_dist"] = np.std(dists)
        metrics["max_dist"] = np.max(dists)
        metrics["total_dist"] = np.sum(dists)

        return metrics



