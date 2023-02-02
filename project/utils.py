from constant import *
import numpy as np
import math
from os import path
import csv
from math import sqrt
from PIL import Image, ExifTags

import cv2


import pdb


# COLOR INFORMATION
# (0, 0, 255) is RED
# (0, 255, 0) is GREEN


def load_original(file_name, dir):
    # mosaic_dir = os.path.join(Path.cwd().parent, "data/mosaic/")
    # first_file = os.path.join(constant.MOSAIC_DIR, file_name)
    # print(first_file)
    # print(f"The mosaic directory is : {constant.MOSAIC_DIR}")
    # print(f"The other mosaic directory is : {constant.MOSAIC_DIR_OTHER}")
    # print(MOSAIC_DIR)
    if dir == "source":
        file_path = os.path.join(SOURCE_DIR, file_name)
    elif dir == "contoured":
        file_path = os.path.join(CONTOURED_DIR, file_name)
    elif dir == "cropped":
        file_path = os.path.join(CROPPED_DIR, file_name)
    original = cv2.imread(file_path)
    return original


def load_metadata(filename):
    """
    Designed to load 2 types of files in dictionnary
    pictures_per_mosaic.csv
    rotation_metadata.csv
    """
    with open(filename, mode="r") as file:
        reader = csv.reader(file)
        next(reader, None)
        # if len(rows) > 1 : prevents bug when additional carriage return
        # at the end of the metadata file
        mapping = {rows[0]: int(rows[1]) for rows in reader if len(rows) > 1}
    return mapping


def show(title, image):
    cv2.imshow(title, image)
    cv2.waitKey()
    cv2.destroyAllWindows()
    cv2.waitKey(1)


def draw(img, contour, color_index=0, show_points=False, show_index=False, legend: list = []):
    """
    This function DOES impact img - it simply display on top of a duplicate
    Font scale of 4 is good (8 is way too big)

    Contour HAS TO BE A SINGLE CONTOUR.
    Because we're then iterating through the points of that contour
    """

    OFFSET_HORIZONTAL = 20
    OFFSET_VERTICAL = 20

    num_row, num_col = img.shape[:2]
    mid_left_right = int(num_col / 2)  # determines
    mid_top_down = int(num_row / 2)

    def start_text_location(point: tuple):
        point_x = point[0]
        point_y = point[1]
        if point_x <= mid_left_right and point_y < mid_top_down:
            # Top left section
            start_position = (point_x + OFFSET_HORIZONTAL, point_y + OFFSET_VERTICAL + 100)
        elif point_x <= mid_left_right and point_y >= mid_top_down:
            # Bottom left section
            start_position = (point_x + OFFSET_HORIZONTAL, point_y - OFFSET_VERTICAL)
        elif point_x > mid_left_right and point_y < mid_top_down:
            # Top right section
            start_position = (point_x - OFFSET_HORIZONTAL - 1000, point_y + OFFSET_VERTICAL + 100)
        elif point_x > mid_left_right and point_y >= mid_top_down:
            # Bottom right section
            start_position = (point_x - OFFSET_HORIZONTAL - 1000, point_y - OFFSET_VERTICAL)
        return start_position

    if legend is None:
        legend = []
    # img_copy = img.copy()
    # pdb.set_trace()
    cv2.drawContours(img, [contour], -1, COLOR_LIST[color_index], CONTOUR_SIZE)
    if show_points:
        for idx, point in enumerate(contour):
            f_point = point.flatten()
            # Enables to handle different versions of point
            # point can be [[345, 277]], or [345, 277]. tuple(xxx) would not work in the 1st config
            cv2.circle(img, center=tuple(f_point), radius=20, color=POINT_COLOR, thickness=cv2.FILLED)
            if show_index:
                cv2.putText(img, f"{idx} - {tuple(f_point)}", start_text_location(f_point), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 12, cv2.LINE_AA)
            if len(legend) > 0:
                cv2.putText(img, f"Legend: {legend[idx]}", (5 + f_point[0], 150 + f_point[1]), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 12, cv2.LINE_AA)


def stack_images(list_labels, list_images, message, num_columns=4):
    """
    num_horizontal : number of horizontal pictures
    list_labels = ['Original', 'Grey', 'Blur', 'Threshold', 'Contours']
    """

    SUMMARY_PAGE = "Results - Summary"
    num_images = len(list_images)
    num_labels = len(list_labels)
    if num_images != num_labels:
        print("Error - Lists of different sizes")
    else:
        list_labels.append(SUMMARY_PAGE)
        num_labels = len(list_labels)
        num_rows = math.ceil(num_labels / num_columns)
        # print(f"Number of rows : {num_rows}")
        # num_rows = int(num_images / num_columns) + 1 # This creates a corner case when num_images = 4. It would go to 8 total_images
        total_num_images = num_columns * num_rows
        num_blank_images = total_num_images - num_images

        # Size of original image
        img_num_rows, img_num_col, num_channels = list_images[0].shape

        # List contains images we want to display
        list_images_multidim = []

        for img in list_images:
            if len(img.shape) == 3:
                list_images_multidim.append(img)
            else:
                multidim = np.stack((img,) * 3, axis=-1)
                list_images_multidim.append(multidim)

        blank_image = np.full((img_num_rows, img_num_col, 3), 255, np.uint8)
        # Create a list of num_horizontal blank images
        # Which will be replaced by an actual picture 1 by 1
        # With default values : it has 4 blank images

        # blank_hor_stack = [blank_image]*num_horizontal
        # Adding 1 blank image for the summary
        blank_images = [blank_image] * num_blank_images
        blank_labels = ["Blank Image"] * num_blank_images
        # List contains images we want to display + blank images

        all_images = list_images_multidim.copy()
        all_images.extend(blank_images)

        all_labels = list_labels.copy()
        all_labels.extend(blank_labels)

        all_images_reorged = [all_images[i : i + num_columns] for i in range(0, len(all_images), num_columns)]

        horizontals = []

        for horizontal_set in all_images_reorged:
            horizontals.append(np.hstack(horizontal_set))

        if len(horizontals) >= 1:
            final = np.vstack(horizontals)
        else:
            final = horizontals[0]

        # USELESS SECTION SINCE WE DON'T USE THE STRING
        # Restructuring the message

        final_message_string = ""

        for item in message.items():
            final_message_string += f"{item[0]} : {item[1]}\n"

        # END OF USELESS SECTION

        final_message_columns = [
            "total_num_contours",
            "num_biggest_contours",
            "num_rectangles_before_split",
            "photos_areas",
            "split_contours",
            "true_num_pictures",
            "num_contours_after_split",
            "num_points_per_contour",
            "success",
        ]

        new_message = {k: message[k] for k in final_message_columns}
        new_message["modifs"] = f"rm_black_edges : {message['rm_black_edges']} | add_white_margin : {message['add_white_margin']}"
        new_message["blur"] = f"blur_method : {message['blur_method']} | blur_parameters : {message['blur_parameters']}"
        new_message["thresh"] = f"threshold_method  : {message['threshold_method']} | min :  {message['threshold_min']} | max :  {message['threshold_max']}"

        print(f"Contouring succeeded : {new_message['success']}")
        # Labeling

        for x in range(num_columns):
            for y in range(num_rows):
                index_picture = x + y * num_columns
                image_name = all_labels[index_picture]
                cv2.putText(
                    final,
                    f"{index_picture + 1} - {image_name}",
                    (5 + img_num_col * x, 500 + img_num_rows * y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    8,
                    (0, 0, 255),
                    # (0, 255, 0),
                    12,
                    cv2.LINE_AA,
                )

                if image_name == SUMMARY_PAGE:
                    starting_point = 1000
                    i = 0
                    gap = 400
                    # Inspired by https://gist.github.com/imneonizer/b64cdd8e2dc23451f5d8caf8279b3ff5
                    for item in new_message.items():
                        cv2.putText(
                            final,
                            f"{item[0]} : {item[1]}",
                            (5 + img_num_col * x, 1000 + i * gap + img_num_rows * y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            8,
                            (0, 0, 0),
                            12,
                            cv2.LINE_AA,
                        )
                        # print(f"Writing at coordinates {5 + img_num_col * x} - {1000 + i * gap + img_num_rows * y}")
                        i += 1

    return final


def config_num():
    RESULTS_PATH = PROJECT_DIR + "results/"
    csv_file = RESULTS_PATH + "results.csv"
    if path.exists(csv_file):
        with open(csv_file, "r") as file:
            csvread = csv.reader(file)
            lines = []
            for row in csvread:
                # print(row)
                lines.append(row)
            max_config_num = int(lines[-1][-1])
            # print(max_config_num)
            current_config_nun = max_config_num + 1
    else:
        current_config_nun = 1
    return current_config_nun


def log_results(message: dict, result_file_name):
    RESULTS_PATH = PROJECT_DIR + "results/"
    csv_file = RESULTS_PATH + result_file_name

    new_message = message.copy()
    csv_headers = list(new_message.keys())
    # print(csv_headers)
    # writer = csv.DictWriter(w, fieldnames=csv_headers)
    num_rows_to_insert = len(new_message["config_num"])
    print(num_rows_to_insert)
    if path.exists(csv_file) is False:
        with open(csv_file, "a") as wr:
            writ = csv.writer(wr)
            print("Printing the headers")
            writ.writerow(csv_headers)

    with open(csv_file, "a") as w:
        writer = csv.writer(w)
        for i in range(num_rows_to_insert):
            writer.writerow([new_message[x][i] for x in csv_headers])


def get_point_density(points):
    total_distance = 0
    count = 0
    i = 0
    for x1, y1 in points:
        for x2, y2 in points[i + 1 :]:
            count += 1
            # print(f"distance between ({x1},{y1}) and ({x2},{y2})")
            distance = sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            total_distance += distance
        i += 1
    return count / total_distance


def rotate_exif(filepath):
    """
    Rotate image based on metadata stored about it

    Args:
        filepath: File location

    Returns:
        Nothing. It just rotates the picture and overwrites the image

    Sources:
        - https://stackoverflow.com/questions/13872331/rotating-an-image-with-orientation-specified-in-exif-using-python-without-pil-in
    """

    try:
        image = Image.open(filepath)
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == "Orientation":
                break
        exif = dict(image._getexif().items())

        if exif[orientation] == 3:
            image = image.transpose(Image.ROTATE_180)
        elif exif[orientation] == 6:
            image = image.transpose(Image.ROTATE_270)
        elif exif[orientation] == 8:
            image = image.transpose(Image.ROTATE_90)
        image.save(filepath)
        image.close()

    except (AttributeError, KeyError, IndexError) as e:
        print(f"Error : {e}")  # cases: image don't have getexif
        pass


def nothing(x):
    pass


def white_triangle(mosaic_name="mamie0022.jpg", triangle_length=WHITE_TRIANGLE_LENGTH, triangle_height=WHITE_TRIANGLE_HEIGHT, color=(255, 255, 255)):
    """
    Very specific to my use case
    """
    img = load_original(mosaic_name, dir="source")
    img_whiten_edges = img.copy()
    num_row, num_col = img_whiten_edges.shape[:2]
    top_left = (0, 0)
    bottom_right_vertical = (THICKNESS_VERTICAL, num_row)
    bottom_right_horizontal = (num_col, THICKNESS_HORIZONTAL)
    cv2.rectangle(img_whiten_edges, top_left, bottom_right_vertical, color, -1)
    cv2.rectangle(img_whiten_edges, top_left, bottom_right_horizontal, color, -1)
    # img = load_original(file_name="mamie0171.jpg", dir="source")
    # img = load_original(file_name=file_name, dir="source")

    img_triangle = img_whiten_edges.copy()
    num_rows, num_columns = img_triangle.shape[:2]

    # Most left point of the triangle (tip is at -400 on X axis)
    point_1 = (num_columns - WHITE_TRIANGLE_LENGTH, THICKNESS_HORIZONTAL + 1)
    # Top right point
    point_2 = (num_columns - 1, THICKNESS_HORIZONTAL + 1)
    # Bottom right point
    point_3 = (num_columns - 1, THICKNESS_HORIZONTAL + 1 + WHITE_TRIANGLE_HEIGHT)

    triangle_cnt = np.array([point_1, point_2, point_3])

    cv2.drawContours(img_triangle, [triangle_cnt], 0, color, -1)

    filename = mosaic_name.split(".")[0]
    cv2.imwrite(f"{filename}_img_whiten_edges.jpg", img_whiten_edges)
    cv2.imwrite(f"{filename}_img_triangle.jpg", img_triangle)
    return img_whiten_edges, img_triangle


def draw_point(img, point: tuple, show_image=True, write=True):
    """
    img.shape returns height (num_rows / Y), width (num_columns / X), channels
    """
    # copy = img.copy()
    copy = img  # Actually writing on the image
    color = (0, 0, 255)
    cv2.circle(copy, point, radius=5, color=color, thickness=cv2.FILLED)
    if show_image:
        show("Position of pixel", copy)
    if write:
        cv2.imwrite("point_pixel.jpg", copy)
    return copy


def log_coordinates(event, x, y, flags, params):
    coords = []
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"x : {x} - y : {y}")
        print(f"({x},{y})")
        print(f"Flags : {flags}")
        print(f"Params : {params}")
        coords.append((x, y))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + "," + str(y), (x, y), font, 5, (255, 0, 0), 2)
        cv2.imshow("Picture to investigate", img)


def find_black_edge(mosaic_name="mamie0022.jpg", color=(255, 255, 255)):
    img = load_original(mosaic_name, dir="source")
    img_whiten_edges = img.copy()
    num_row, num_col = img_whiten_edges.shape[:2]
    top_left = (0, 0)
    bottom_right_vertical = (THICKNESS_VERTICAL, num_row)
    bottom_right_horizontal = (num_col, THICKNESS_HORIZONTAL)
    cv2.rectangle(img_whiten_edges, top_left, bottom_right_vertical, color, -1)
    cv2.rectangle(img_whiten_edges, top_left, bottom_right_horizontal, color, -1)


def initializeTrackbars(intialTracbarVals=0, threshold=True):
    cv2.namedWindow("Trackbars Window")
    if threshold:
        default_min_thresh = 230
        slider_max = 255
        cv2.createTrackbar("Threshold Min", "Trackbars Window", default_min_thresh, slider_max, nothing)


def valTrackbars():
    min_thresh = cv2.getTrackbarPos("Threshold Min", "Trackbars Window")
    src = min_thresh
    return src


if __name__ == "__main__":

    # BELOW IS THE CODE TO DETERMINE THE AREA OF A CONTOUR POINTED WITH THE MOUSE
    # TO BETTER DEFINE THE MIN_AREA THRESHOLD

    # "mamie0450.jpg" fails on top left, top right, bottom left pictures
    # "mamie0452.jpg" fails on all pictures
    # "mamie0453.jpg" fails on all pictures
    # "mamie0458.jpg" fails on all pictures
    # "mamie0462.jpg" fails on all pictures - very little ones - check the one in the middle

    # https://www.geeksforgeeks.org/displaying-the-coordinates-of-the-points-clicked-on-the-image-using-python-opencv/
    # file_name = "mamie0458.jpg"

    file_name = "mamie0462.jpg"
    img = load_original(file_name, dir="source")
    cv2.imshow("Picture to investigate", img)
    cv2.setMouseCallback("Picture to investigate", log_coordinates)
    cv2.waitKey()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    # import numpy as np
    # import cv2
    # from utils import *

    # The list below is for "mamie0458.jpg" - Area = 4901879.0
    # list_points = [(1255, 165), (3770, 187), (3746, 2120), (1231, 2130)]

    # The list below is for "mamie0462.jpg" - Area = 2940480.5
    # list_points = [(1584,2192), (3035,2171), (3099,4189), (1662,4243)]

    list_points = [(1584, 2192), (3035, 2171), (3099, 4189), (1662, 4243)]
    contour = np.array(list_points, dtype=int)
    # Check that contour is correct
    # file_name = "mamie0458.jpg"
    file_name = "mamie0462.jpg"
    img_copy = load_original(file_name, dir="source")
    cv2.drawContours(img_copy, [contour], -1, (0, 255, 0), CONTOUR_SIZE)
    show("Test contour", img_copy)
    contour_area = cv2.contourArea(contour)
    print(contour_area)
    # Contour Area is below threshold for picture in
    contour_area < MIN_AREA_THRESHOLD

    """
    # Source : https://www.tutorialspoint.com/opencv-python-how-to-display-the-coordinates-of-points-clicked-on-an-image
    while True:
        cv2.imshow("Picture to investigate", img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    # print(coords)
    """

    """
    # BELOW IS THE CODE TO IDENTIFY THE LOCATION OF THE PROBLEMATIC BLACK EDGE
    
    
    # from utils import *
    mosaic_name = "mamie0193.jpg"
    color = (255, 255, 255)
    img = load_original(mosaic_name, dir="source")
    img_whiten_edges = img.copy()
    num_row, num_col = img_whiten_edges.shape[:2]
    top_left = (0, 0)
    bottom_right_vertical = (THICKNESS_VERTICAL, num_row)
    bottom_right_horizontal = (num_col, THICKNESS_HORIZONTAL)
    cv2.rectangle(img_whiten_edges, top_left, bottom_right_vertical, color, -1)
    cv2.rectangle(img_whiten_edges, top_left, bottom_right_horizontal, color, -1)
    max_y, max_x = img_whiten_edges.shape[:2]

    # 600 is not bad, starting to get close to the edge of the actual Picture
    # draw_point(img_whiten_edges, (max_x - 600, THICKNESS_HORIZONTAL), show_image=True, write=False)

    img_whiten_edges_copy = img_whiten_edges.copy()
    # X_OFFSET = 600
    X_OFFSET = 40
    for i in range(15):
        point = (max_x - X_OFFSET, THICKNESS_HORIZONTAL - 4 + i)
        # array[row_number:col_number]. So actually written as array[Y:X]
        # print(f"Point position : X = {point[0]}, Y = {point[1]}, BGR = {img_whiten_edges[point[0], point[1]]}")
        print(f"Point position : X = {point[0]}, Y = {point[1]}, BGR = {img_whiten_edges[point[1], point[0]]}")
        draw_point(img_whiten_edges_copy, point, show_image=True, write=False)
    draw_point(img_whiten_edges_copy, point, show_image=True, write=True)

    """
    pass
