import numpy as np


class Contour:
    def __init__(self):
        pass

    def get_angles(self):
        self.shape = (-1, 2)
        a = self - np.roll(self, 1, axis=0)
        b = np.roll(a, -1, axis=0)
        alengths = np.linalg.norm(a, axis=1)
        blengths = np.linalg.norm(b, axis=1)
        crossproducts = np.cross(a, b) / alengths / blengths
        angles_radians = np.arcsin(crossproducts)
        angles = angles_radians / np.pi * 180
        self.angles = angles  # in degrees
        self.alengths = alengths  # Length of the segment on 1 side of the angle
        self.blengths = blengths  # Lenght of the other segment on the other side of the angle

        return angles, alengths, blengths
