import cv2
from constant import *
from Mosaic import *
from Contour import *
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
    # drawing contours is done on the modified image with added borders
    img_w_contours = mosaic.img_source.copy()
    cv2.drawContours(img_w_contours, mosaic.contours, -1, (0, 255, 0), 30)
    # print("Number of contours identified: ", len(contours))
    if show_image:
        show("Original with contours", img_w_contours)

    return img_w_contours


def draw_main_contours(
    mosaic: Mosaic,
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

    # Drawing is done on the img with borders
    img_w_main_contours = mosaic.img_source.copy()

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

    return message


def fix_contours(mosaic):
    final_image = mosaic.img_source.copy()
    contours_final = []
    color_index = 0

    for elem in mosaic.contours_main:
        # If need to print or draw : do it on elem
        contour = Contour(elem)
        cv = contour.plot_points(show=False)
        if contour.scission_point is not None:
            # print("Contour needs to be splitted")
            contour.find_extrapolation()
            split_contours, intersection_point = contour.split_contour(mosaic.img_source, cv)
            new_contours = split_contours
            show("cv", cv)
        else:
            # print(f"Contour has good shape - no need for split - color index = {color_index}")
            # new_contours has to be a list - in this case, it's a list of 1 single element
            # from_enriched_to_regular is necessary : it removes the bad points
            # and takes advantaged of the cleaning done in enrich_contour()
            clean_contour = from_enriched_to_regular(contour.enriched)
            new_contours = [clean_contour]
        for cont in new_contours:
            contours_final.append(cont)
            draw(final_image, cont, color_index)
            color_index += 1
    mosaic.contours_final = contours_final
    mosaic.num_contours_final = len(contours_final)
    mosaic.img_w_final_contours = final_image
    # UNCOMMENT FOR TESTING
    show("Final contours", final_image)

    return contours_final, final_image


if __name__ == "__main__":
    pass
