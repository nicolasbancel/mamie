import cv2
from constant import *
from Mosaic import *
from utils import *
import pdb


CONTOUR_SIZE = 40
CONTOUR_COLOR_DEFAULT = (0, 255, 0)
CONTOUR_PRECISION_PARAM = 0.01


def find_contours(mosaic, retrieval_mode=cv2.RETR_EXTERNAL):
    # cv2.RETR_EXTERNAL : external contours only
    contours, hierarchy = cv2.findContours(mosaic.img_thresh.copy(), retrieval_mode, cv2.CHAIN_APPROX_NONE)
    mosaic.contours_all = contours
    return contours, hierarchy


def draw_contours(mosaic, show_image=False):
    img_w_contours = mosaic.img.copy()
    cv2.drawContours(img_w_contours, mosaic.contours, -1, (0, 255, 0), 30)
    # print("Number of contours identified: ", len(contours))
    if show_image:
        show("Original with contours", img_w_contours)

    return img_w_contours


def reshape_contour(contour):
    pass


def draw_main_contours(
    mosaic,
    contour_size=CONTOUR_SIZE,
    contours_color=CONTOUR_COLOR_DEFAULT,
    precision_param=CONTOUR_PRECISION_PARAM,
    only_rectangles=None,
    show_image=None,
):
    """
    This does the rotation for 1 picture, a list of hardcoded picture, or n pictures in the CROPPED folder

    Args:
        mosaic :
        contour_size : :   If not None, will execute the rotation for the [0:num_pictures] in the cropped folder
        contours_color :   If True, fills and pushes the logs to the results_rotation.csv file
        precision_param :  precision_param=0.01 : very precise. Creates sharp angles, and prevents from identifying simple rectangles sometimes
                           https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
                           precision_param=0.1 : should approximate much more rectangles.
                           Although, it would prevent from splitting the "big picture in 2"
        only_rectangles:   If True: would focus / draw only on polygons with 4 sides
        show_image:        If True : shows the image

    Returns :
    """

    # no_approx_main_contours : has no shape approximation
    no_approx_main_contours = sorted([c for c in mosaic.contours_all if cv2.contourArea(c) > MIN_AREA_THRESHOLD], key=cv2.contourArea, reverse=True)

    contours_main = []
    num_rectangles = 0

    for c in no_approx_main_contours:
        ### Approximating the contour
        # Calculates a contour perimeter or a curve length
        peri = cv2.arcLength(c, True)

        approx = cv2.approxPolyDP(c, precision_param * peri, True)
        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        screenCnt = approx
        # print(f'Number of corners : {len(approx)}')
        if len(approx) == 4:
            num_rectangles += 1
        if only_rectangles:
            if len(approx) == 4:
                contours_main.append(screenCnt)
        else:
            contours_main.append(screenCnt)
        # show the contour (outline)

    img_w_main_contours = mosaic.img.copy()

    contours_areas = [cv2.contourArea(x) for x in contours_main]

    cv2.drawContours(img_w_main_contours, contours_main, -1, contours_color, contour_size)

    mosaic.contours_main = contours_main
    mosaic.img_w_main_contours = img_w_main_contours
    mosaic.num_contours_total = len(mosaic.contours_all)
    mosaic.num_contours_main = len(contours_main)

    message = {
        "total_num_contours": len(mosaic.contours_all),
        "num_biggest_contours": len(contours_main),
        "num_rectangles_before_split": num_rectangles,
        "photos_areas": contours_areas,
    }

    # print(message)
    # print(f"Out of {num_biggest_contours} biggest contours - {num_rectangles} are rectangles")

    if show_image:
        show("Original w Main Contours", img_w_main_contours)

    return img_w_main_contours, contours_main, message


def from_enriched_to_regular(enriched_contour):
    contour = []

    for point in enriched_contour:
        x = int(point[0])
        y = int(point[1])
        contour.append([x, y])
    contour = np.array(contour, dtype=np.int64)

    return contour


if __name__ == "__main__":
    pass
