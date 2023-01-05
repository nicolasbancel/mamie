import cv2 as cv2
import os
from pathlib import Path

THRESH_MIN = 252  # TESTING VS 250 TO HAVE ONLY THE PHOTOCOPIEUSE WHITE CONSIDERED WHITE
THESH_MAX = 255

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


MOSAIC_DIR = "/Users/nicolasbancel/git/perso/mamie/data/mosaic/"
# MOSAIC_DIR = os.path.join(Path.cwd().parent, "data/mosaic/")
# print(f"The mosaic directory is : {MOSAIC_DIR}")
# MOSAIC_DIR_OTHER = os.path.join(Path(os.path.abspath(os.curdir)).parent, "data/mosaic/")
# print(f"The other mosaic directory is : {MOSAIC_DIR_OTHER}")
