import os

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from src.utils.Queries import QUERIES
from src.utils.Colors import COLORS
from src.utils.Utils import download_images

images_path = "../resources/images/"

def get_template(board_type, name, description):
    folder = images_path + name + "_" + description
    if  not os.path.exists(folder):
        os.makedirs(folder)
    if  not os.listdir(folder):
        print(f"No images for {board_type} board {name} with description {description}, downloading them now ...")
        download_images(board_type, name, description)
        print("Images downloaded successfully")
    image_path = os.path.join(folder, os.listdir(folder)[0])
    image_merge = Image.open(image_path).convert("RGBA")
    for file in os.listdir(folder):
        image_path = os.path.join(folder, file)
        image_merge = Image.alpha_composite(image_merge, Image.open(image_path).convert("RGBA"))
    return image_merge



def make_circle(color, size=50):
    """
    Draws a circle with center (x, y) on an RGBA image.

    Parameters:
        color (tuple): RGBA color of the circle
        size (tuple): Size of the output image (width, height)

    Returns:
        PIL.Image: Image with the circle drawn
    """
    circle = Image.new("RGBA", (size, size), (0, 0, 0, 0))  # Transparent background
    draw = ImageDraw.Draw(circle)
    left_up = (0, 0)
    right_down = (size, size)
    draw.ellipse([left_up, right_down], outline=color, width=4)
    return circle


def get_xy_for_ids(df: pd.DataFrame, ids: list[int]) -> dict[int, list[tuple[int, int]]]:
    """
    For each integer in the list ids, find the (x, y) pairs
    using the first two columns as references.

    Returns a dictionary: {id: [(x, y), ...]}
    """
    results = {}
    for i in ids:
        subset = df[(df.iloc[:, 0] == i) | (df.iloc[:, 1] == i)]
        results[i] = list(zip(subset.iloc[:, 2], subset.iloc[:, 3]))
    return results

def get_edges(name, connection):
    df = pd.read_sql(QUERIES["edges"], connection, None, None, {'name': name})
    return df["edge_left"].values, df["edge_right"].values, df["edge_bottom"].values, df["edge_top"].values

class Visualizer():
    def __init__(self, board, connection):
        self.board = board
        self.connection = connection
    def visualize_climb(self, matrix):

        start_color = COLORS["start"]
        middle_color = COLORS["middle"]
        finish_color = COLORS["finish"]
        foot_color = COLORS["foot"]

        edge_left, edge_right, edge_bottom, edge_top = self.board.get_edges()
        # Getting the board image without markers
        template = self.board.template
        width, height = template.size

        xSpacing = width / (edge_right - edge_left)
        ySpacing = height / (edge_top - edge_bottom)

        for i in range(len(matrix)): # i is the Y coordinate
            for j in range(len(matrix[i])): # j is the X coordinate
                if matrix[i][j] > 0 :
                    xPixel = int((i - edge_left) * xSpacing)
                    yPixel = int(height - (j - edge_bottom) * ySpacing)
                    color = "#000000"
                    if matrix[i][j] == 1: # start hold
                        color = start_color
                    if matrix[i][j] == 2:
                        color = middle_color
                    if matrix[i][j] == 3:
                        color = finish_color
                    if matrix[i][j] == 4:
                        color = foot_color
                    circle = make_circle(color)
                    template.paste(circle, (xPixel - circle.width // 2, yPixel- circle.height // 2), circle)
        return template
















