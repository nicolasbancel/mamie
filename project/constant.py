import cv2 as cv2
import os
from pathlib import Path


THICKNESS_HORIZONTAL = 25
THICKNESS_VERTICAL = 15
WHITE_TRIANGLE_HEIGHT = 10
WHITE_TRIANGLE_LENGTH = 400


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

FINAL_MESSAGE = {
    "execution_time": [],
    "total_num_contours": [],
    "num_biggest_contours": [],
    "num_rectangles_before_split": [],
    "photos_areas": [],
    "success": [],
    "picture_name": [],
    "rm_black_edges": [],
    "add_white_margin": [],
    "blur_method": [],
    "blur_parameters": [],
    "threshold": [],
    "threshold_method": [],
    "threshold_min": [],
    "threshold_max": [],
    "split_contours": [],
    "true_num_pictures": [],
    "num_contours_after_split": [],
    "num_points_per_contour": [],
    "config_num": [],
}


FINAL_LOG_CONTOURS = {
    "execution_time": [],
    "total_num_contours": [],
    "num_biggest_contours": [],
    "num_rectangles_before_split": [],
    "photos_areas": [],
    "success": [],
    "picture_name": [],
    "rm_black_edges": [],
    "add_white_margin": [],
    "blur_method": [],
    "blur_parameters": [],
    "threshold": [],
    "threshold_method": [],
    "threshold_min": [],
    "threshold_max": [],
    "split_contours": [],
    "true_num_pictures": [],
    "num_contours_after_split": [],
    "num_points_per_contour": [],
    "config_num": [],
}

FINAL_LOG_ROTATIONS = {
    "config_num": [],
    "picture_name": [],
    "rot90_true_num": [],
    "rot90_predicted_num": [],
    "success": [],
    "rot90_summary": [],
}

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


CONTOUR_SIZE = 20
CONTOUR_COLOR_DEFAULT = (0, 255, 0)
CONTOUR_PRECISION_PARAM = 0.01


# List of colors : BGR
# List of good colors to print contours with
# Green, Red, Blue, Yellow, Cyan, Magenta
COLOR_LIST = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255)]
POINT_COLOR = (0, 0, 0)

PROJECT_DIR = "/Users/nicolasbancel/git/perso/mamie/project/"

SOURCE_DIR = "/Users/nicolasbancel/git/perso/mamie/data/mosaic/source"
CONTOURED_DIR = "/Users/nicolasbancel/git/perso/mamie/data/mosaic/contoured/"
CROPPED_DIR = "/Users/nicolasbancel/git/perso/mamie/data/mosaic/cropped/"
ROTATED_AUTO_DIR = "/Users/nicolasbancel/git/perso/mamie/data/mosaic/rotated_automatic/"
ROTATED_MANUAL_DIR = "/Users/nicolasbancel/git/perso/mamie/data/mosaic/rotated_manual/"
TO_TREAT_DIR = "/Users/nicolasbancel/git/perso/mamie/data/mosaic/to_treat_manually/"

OPENCV_DATA_DIR = "/Users/nicolasbancel/git/perso/mamie/data/opencv/"

#######################################################################
# FACE DETECTION MODELS
#######################################################################

YUNET_PATH = "/Users/nicolasbancel/git/perso/mamie/data/opencv/face_detection_yunet_2022mar.onnx"
SFACE_PATH = "/Users/nicolasbancel/git/perso/mamie/data/opencv/face_recognition_sface_2021dec.onnx"


# MOSAIC_DIR = os.path.join(Path.cwd().parent, "data/mosaic/")
# print(f"The mosaic directory is : {MOSAIC_DIR}")
# MOSAIC_DIR_OTHER = os.path.join(Path(os.path.abspath(os.curdir)).parent, "data/mosaic/")
# print(f"The other mosaic directory is : {MOSAIC_DIR_OTHER}")
