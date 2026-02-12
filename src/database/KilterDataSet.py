import h5py
import numpy as np
import pandas as pd




class KilterDataset:
    def __init__(self, x_coords,y_coords, angles, grades):
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.angles = angles
        self.grades = grades

    def get_x_coords(self):
        return self.x_coords

    def get_y_coords(self):
        return self.y_coords

    def get_angles(self):
        return self.angles

    def get_grades(self):
        return self.grades





        