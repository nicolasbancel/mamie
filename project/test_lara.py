from utils import *
from constant import *


MOSAIC_METADATA = load_metadata(filename="pictures_per_mosaic.csv")


def whiten_edges(
    img,
    thickness_vertical=THICKNESS_VERTICAL,
    thickness_horizontal=THICKNESS_HORIZONTAL,
    color=(0, 0, 0),  # color=(0, 255, 0) for GREEN
    show_image=False,
):
    img_copy = img.copy()
    num_row, num_col = img_copy.shape[:2]
    top_left = (0, 0)
    bottom_right_vertical = (thickness_vertical, num_row)
    bottom_right_horizontal = (num_col, thickness_horizontal)

    # Adding vertical rectangle on the left
    cv2.rectangle(img_copy, top_left, bottom_right_vertical, color, -1)

    # Adding horizontal rectangle on top
    cv2.rectangle(img_copy, top_left, bottom_right_horizontal, color, -1)

    if show_image:
        show("With white edges", img_copy)

    return img_copy


def whiten_triangle(img, triangle_length=WHITE_TRIANGLE_LENGTH, triangle_height=WHITE_TRIANGLE_HEIGHT, color=(0, 0, 0), show_image=None):
    """

    Args:
      Applies to image after addition of vertical and horizontal white edges
      There is a black residual triangle that we'll replace with white
    """
    img_triangle = img.copy()
    num_rows, num_columns = img_triangle.shape[:2]

    # Most left point of the triangle (tip is at -400 on X axis)
    point_1 = (num_columns - WHITE_TRIANGLE_LENGTH, THICKNESS_HORIZONTAL + 1)
    # Top right point
    point_2 = (num_columns - 1, THICKNESS_HORIZONTAL + 1)
    # Bottom right point
    point_3 = (num_columns - 1, THICKNESS_HORIZONTAL + 1 + WHITE_TRIANGLE_HEIGHT)

    triangle_cnt = np.array([point_1, point_2, point_3])

    cv2.drawContours(img_triangle, [triangle_cnt], 0, color, -1)

    if show_image == True:
        show("With white edges + triangle", img_triangle)
    return img_triangle


def add_borders(img, color=(0, 0, 0), show_image=False):
    # This script assumes we're adding borders to an image where the edges have already been blanked
    # It applies to self.img_white_edges, not to self.img
    source = img
    border_size = min(int(0.05 * source.shape[0]), int(0.05 * source.shape[1]))
    top = bottom = left = right = border_size
    borderType = cv2.BORDER_CONSTANT
    img_copy = cv2.copyMakeBorder(source, top, bottom, left, right, borderType, None, color)

    if show_image:
        show("With white border", img_copy)

    return img_copy


def grey_original(img, show_image=False):
    # This script assumes that greying happens after whiten edges + add borders
    img_grey = img.copy()
    img_grey = cv2.cvtColor(img_grey, cv2.COLOR_BGR2GRAY)
    if show_image:
        show("Grey image", img_grey)
    return img_grey


def thresh(img, show_image=False, thresh_min=0, thresh_max=10, method=cv2.THRESH_BINARY):
    """
    This does 2 things :
    - It both blurs the image
    - And it finds the threshold
    """
    img_thresh = cv2.threshold(img, thresh_min, thresh_max, method)[1]
    if show_image:
        show(f"Image Threshold - Min : {thresh_min} - Max : {thresh_max}", img_thresh)
    return img_thresh


def draw_contours(img, contours, show_image=False):
    # drawing contours is done on the modified image with added borders
    img_w_contours = img.copy()
    cv2.drawContours(img_w_contours, mosaic.contours, -1, (0, 255, 0), 30)
    # print("Number of contours identified: ", len(contours))
    if show_image:
        show("Original with contours", img_w_contours)

    return img_w_contours


def draw_main_contours(
    img_white_borders,
    contours,
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
    MIN_AREA_THRESHOLD = 500000  # Decreasing size of the contour
    no_approx_main_contours = sorted([c for c in contours if cv2.contourArea(c) > MIN_AREA_THRESHOLD], key=cv2.contourArea, reverse=True)

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
        if len(approx) == 4:
            num_rectangles += 1
        if only_rectangles:
            if len(approx) == 4:
                contours_main.append(screenCnt)
        else:
            contours_main.append(screenCnt)
        # show the contour (outline)

    # Drawing is done on the img with borders
    img_w_main_contours = img_white_borders.copy()

    contours_areas = [cv2.contourArea(x) for x in contours_main]

    # pdb.set_trace()
    for contour in contours_main:
        draw(img_w_main_contours, contour, color_index=0, show_points=True, show_index=True)
    # cv2.drawContours(img_w_main_contours, contours_main, -1, contours_color, contour_size)
    # print(f"Out of {num_biggest_contours} biggest contours - {num_rectangles} are rectangles")

    if show_image:
        show(f"Original w Main Contours", img_w_main_contours)

    return img_w_main_contours, contours_main


if __name__ == "__main__":
    from Mosaic import *
    from matplotlib import pyplot as plt

    mosaic_name = "lara0001.jpg"
    img = load_original(mosaic_name, dir="source")
    img_white_edges = whiten_edges(img, show_image=True)
    img_white_edgesxtriangle = whiten_triangle(img_white_edges, show_image=True)
    img_white_borders = add_borders(img_white_edgesxtriangle, show_image=True)
    img_grey = grey_original(img_white_borders, show_image=True)
    img_blur = cv2.GaussianBlur(img_grey, (3, 3), 0)
    num_rows, num_columns = img_blur.shape[:2]
    img_thresh = thresh(img_blur, show_image=True))


    # Show histogram
    # https://www.geeksforgeeks.org/opencv-python-program-analyze-image-using-histogram/
    img_blur_copy = img_blur.copy()
    histr = cv2.calcHist([img_blur_copy], [0], None, [256], [0, 256])
    plt.plot(histr)
    plt.show()
    # Background color is between 50 and 60
    # So anything between 50 and 60 should be made black

    # Test replacement of all pixels with 50 to 60, or equal to 0 black to black
    # Otherwise keep as is
    
    custom_thresh = img_blur.copy()
    for y in range(num_rows-1):
        for x in range(num_columns-1):
            if 50 <= custom_thresh[y,x] <= 60 or custom_thresh[y,x] == 0:
                custom_thresh[y,x] = 0
    contours, hierarchy = cv2.findContours(custom_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img_w_main_contours, contours_main = draw_main_contours(
    img_white_borders,
    contours,
    contour_size=CONTOUR_SIZE,
    contours_color=CONTOUR_COLOR_DEFAULT,
    precision_param=CONTOUR_PRECISION_PARAM,
    only_rectangles=None,
    show_image=True
    )

    # below 60 should be black
    img_thresh = thresh(img_blur_copy, thresh_min=60, thresh_max=255, show_image=True)

    step = 20
    thresh_min = []
    for i in range(10):
        thresh_min.append(i * step)
        img_thresh = thresh(img_blur, thresh_min=i * step, thresh_max=255, show_image=True)

    # Between 60 and 80 is good
    img_thresh = thresh(img_blur, thresh_min=70, thresh_max=255, show_image=True)
    contours, hierarchy = cv2.findContours(img_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img_w_main_contours, contours_main = draw_main_contours(
        img_white_borders,
        contours,
        contour_size=CONTOUR_SIZE,
        contours_color=CONTOUR_COLOR_DEFAULT,
        precision_param=CONTOUR_PRECISION_PARAM,
        only_rectangles=None,
        show_image=True,
    )

    no_approx_main_contours = sorted(contours, key=cv2.contourArea, reverse=True)
