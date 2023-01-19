import cv2 as cv2
import os
from pathlib import Path

# THRESH_MIN = 252  # TESTING VS 250 TO HAVE ONLY THE PHOTOCOPIEUSE WHITE CONSIDERED WHITE
# THRESH_MIN = 250 # Should be avoided
# THRESH_MIN = 252  # Identifies the biggest contour of the 2 pictures in 1 massive rectangle

# THRESH_MIN = 240  # GOOD - TO KEEP [RUN #1]
# THRESH_MIN = 252  # TESTING VS 250 TO HAVE ONLY THE PHOTOCOPIEUSE WHITE CONSIDERED WHITE

# THRESH_MIN = 250 # TEST RUN #2

THRESH_MIN = 245  # Is what most of my tests were run with
THESH_MAX = 255

# PREVIOUS VALUES :
# SMALL_ANGLE_THRESH = 7 : would not identify one of the small angles in mamie0046.jpg as small angle
SMALL_ANGLE_THRESH = 15
THRESHOLD = 0.25
MAX_AREA_THRESHOLD = 10000000

# MIN_AREA_THRESHOLD = 6000000
# WAS CAUSING PROBLEMS FOR POLAROIDs - THEY WERE TOO SMALL
MIN_AREA_THRESHOLD = 5000000


THRESHOLD_NUM_POINTS_PER_CONTOUR = 5
# Because scission and because duplicate at the beginning and end

morph_size = 9
morph_op_dic = {
    0: cv2.MORPH_OPEN,
    1: cv2.MORPH_CLOSE,
    2: cv2.MORPH_GRADIENT,
    3: cv2.MORPH_TOPHAT,
    4: cv2.MORPH_BLACKHAT,
}
morph_elem_dic = {0: cv2.MORPH_RECT, 1: cv2.MORPH_CROSS, 2: cv2.MORPH_ELLIPSE}

morph_operator = morph_op_dic[0]
morph_elem = morph_op_dic[0]


###


PROJECT_DIR = "/Users/nicolasbancel/git/perso/mamie/project/"

MOSAIC_DIR = "/Users/nicolasbancel/git/perso/mamie/data/mosaic/source"
CONTOURED_DIR = "/Users/nicolasbancel/git/perso/mamie/data/mosaic/contoured/"
# MOSAIC_DIR = os.path.join(Path.cwd().parent, "data/mosaic/")
# print(f"The mosaic directory is : {MOSAIC_DIR}")
# MOSAIC_DIR_OTHER = os.path.join(Path(os.path.abspath(os.curdir)).parent, "data/mosaic/")
# print(f"The other mosaic directory is : {MOSAIC_DIR_OTHER}")
