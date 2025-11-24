import h5py
import numpy as np
import pandas as pd




class KilterDataset:
    def __init__(self, matrices, angles, grades):
        self.matrices = matrices
        self.angles = angles
        self.grades = grades

    def get_matrices(self):
        return self.matrices

    def get_angles(self):
        return self.angles

    def get_grades(self):
        return self.grades





        