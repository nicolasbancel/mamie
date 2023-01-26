from constant import *
import numpy as np
import math
from os import path
import csv
from math import sqrt
from PIL import Image, ExifTags

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
        mapping = {rows[0]: int(rows[1]) for rows in reader}
    return mapping


def show(title, image):
    cv2.imshow(title, image)
    cv2.waitKey()
    cv2.destroyAllWindows()
    cv2.waitKey(1)


def draw(img, contour, color_index=0, show_points=True, show_index=False, legend: list = []):
    """
    This function DOES impact img - it simply display on top of a duplicate
    Font scale of 4 is good (8 is way too big)
    """
    if legend is None:
        legend = []
    # img_copy = img.copy()
    cv2.drawContours(img, [contour], -1, COLOR_LIST[color_index], 40)
    if show_points:
        for idx, point in enumerate(contour):
            cv2.circle(img, center=tuple(point), radius=20, color=POINT_COLOR, thickness=cv2.FILLED)
            if show_index:
                cv2.putText(img, f"{idx} - {tuple(point)}", (5 + point[0], 5 + point[1]), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 12, cv2.LINE_AA)
            if len(legend) > 0:
                cv2.putText(img, f"Legend: {legend[idx]}", (5 + point[0], 150 + point[1]), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 12, cv2.LINE_AA)


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

        print(f"Contouring succeeded : new_message['success']")
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
    print(csv_headers)
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


def find_shape_black_edge_printer():
    """
    Very specific to my use case
    """
    img = load_original(file_name="mamie0171.jpg", dir="source")
    num_rows, num_columns = img.shape[:2]
    color_vertical = (255, 255, 255)
    color_horizontal = (255, 255, 255)
    # color_vertical=(0, 255, 0) # Green
    # color_horizontal=(0, 0, 255) # Red
    top_left = (0, 0)
    bottom_right_vertical = (15, num_rows)
    bottom_right_horizontal = (num_columns, 25)
    # Vertical rectangle
    cv2.rectangle(img, (0, 0), bottom_right_vertical, color_vertical, -1)
    # Horizontal rectangle
    cv2.rectangle(img, (0, 0), bottom_right_horizontal, color_horizontal, -1)
    show("Img with edges", img)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    show("Img with edges in B&W", grey)
    # We should be left only with a tiny black line on the right
    copy = img.copy()
    # img[row, column] - Hence accessing pixel is by img[Y, X] - which is different from circle
    # circle: (X,Y)

    img_without_top_edge = grey[25 + 1 : num_rows, 0:num_columns]
    show("Img without top edge", img_without_top_edge)

    for y in range(30):
        print(f"Column (X) : {num_columns-1} // Row (Y) : {y} // Intensity={img_without_top_edge[y,num_columns-1]}")
        # cv2.circle(copy, (num_columns-1, i), radius=50, color=(0, 255, 0), thickness=-1)
    show("Copy", copy)

    # show the array of last 30 pixels horizontally and until index 7 INCLUDED vertically

    array = img_without_top_edge[0:8, num_columns - 50 : num_columns - 1]
    transp = np.transpose(array)

    X = int(num_columns / 2)

    for i in range(50):
        print(f"Column (X) : {X} // Row (Y) : {i} // Intensity={grey[i,X]}")
        cv2.circle(copy, (X, i), radius=30, color=(255, 0, 0), thickness=-1)
    show("Copy", copy)

    TRIANGLE_HEIGHT = 4
    num_rows, num_columns = img_without_top_edge.shape[:2]
    # Most left point of the triangle (tip is at -400 on X axis)
    point_1 = (num_columns - 400, 0)
    # Top right point
    point_2 = (num_columns - 1, 0)
    # Bottom right point
    point_3 = (num_columns - 1, TRIANGLE_HEIGHT)

    cop = img_without_top_edge.copy()

    cv2.circle(cop, point_1, 50, (0, 0, 255), -1)
    cv2.circle(cop, point_2, 50, (0, 0, 255), -1)
    cv2.circle(cop, point_3, 50, (0, 0, 255), -1)

    show("xx", cop)

    THICKNESS_HORIZONTAL = 25

    img = load_original(file_name="mamie0171.jpg", dir="source")
    num_rows, num_columns = img.shape[:2]
    color_vertical = (255, 255, 255)
    color_horizontal = (255, 255, 255)
    # color_vertical=(0, 255, 0) # Green
    # color_horizontal=(0, 0, 255) # Red
    top_left = (0, 0)
    bottom_right_vertical = (15, num_rows)
    bottom_right_horizontal = (num_columns, THICKNESS_HORIZONTAL)
    # Vertical rectangle
    cv2.rectangle(img, (0, 0), bottom_right_vertical, color_vertical, -1)
    # Horizontal rectangle
    cv2.rectangle(img, (0, 0), bottom_right_horizontal, color_horizontal, -1)

    ## Adding borders
    border_size = min(int(0.05 * img.shape[0]), int(0.05 * img.shape[1]))
    top = bottom = left = right = border_size
    borderType = cv2.BORDER_CONSTANT
    img_borders = cv2.copyMakeBorder(img, top, bottom, left, right, borderType, None, (255, 255, 255))

    # CREATING TRIANGLE
    # Start at 26 is good (THICKNESS_HORIZONTAL + 1)
    # But the height should be a bit higher (5)
    TRIANGLE_HEIGHT = 6
    # color = (200, 200, 200)
    color = (255, 255, 255)
    STARTING_X = 400
    img_triangle = img.copy()
    # Most left point of the triangle (tip is at -400 on X axis)
    point_1 = (num_columns - STARTING_X, THICKNESS_HORIZONTAL + 1)
    # Top right point
    point_2 = (num_columns - 1, THICKNESS_HORIZONTAL + 1)
    # Bottom right point
    point_3 = (num_columns - 1, THICKNESS_HORIZONTAL + 1 + TRIANGLE_HEIGHT)

    triangle_cnt = np.array([point_1, point_2, point_3])

    cv2.drawContours(img_triangle, [triangle_cnt], 0, color, -1)
    cv2.imwrite("img_triangle_L400_H6_start26.jpg", img_triangle)


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
    pass
