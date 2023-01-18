from transfo import *
from contour import *
from testing import *
from constant import *

# import cv2
import numpy as np
import math
from os import path
import csv

import pdb


# COLOR INFORMATION
# (0, 0, 255) is RED
# (0, 255, 0) is GREEN


def num_pictures_per_mosaic(filename="pictures_per_mosaic.csv"):
    with open(filename, mode="r") as file:
        reader = csv.reader(file)
        next(reader, None)
        mapping = {rows[0]: rows[1] for rows in reader}
    return mapping


def draw(img, contour, color_index=0, show_points=True):
    """
    This function does NOT impact img - it simply display on top of a duplicate
    """
    # List of colors : BGR
    # Green, Red, Blue, Yellow, Cyan, Magenta
    color_list = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255)]
    point_color = (0, 0, 0)

    # img_copy = img.copy()
    cv2.drawContours(img, [contour], -1, color_list[color_index], 40)
    if show_points:
        for point in contour:
            cv2.circle(img, center=tuple(point), radius=20, color=point_color, thickness=cv2.FILLED)


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
            # "predicted_num_pictures_on_mosaic",
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

        print(new_message)
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


def log_results(message: dict):
    RESULTS_PATH = PROJECT_DIR + "results/"
    csv_file = RESULTS_PATH + "results.csv"

    new_message = message.copy()
    csv_headers = list(new_message.keys())
    # writer = csv.DictWriter(w, fieldnames=csv_headers)
    num_rows_to_insert = len(new_message["config_num"])
    if path.exists(csv_file) is False:
        with open(csv_file, "a") as wr:
            writ = csv.writer(wr)
            print("Printing the headers")
            writ.writerow(csv_headers)

    with open(csv_file, "a") as w:
        writer = csv.writer(w)
        for i in range(num_rows_to_insert):
            writer.writerow([new_message[x][i] for x in csv_headers])


def nothing(x):
    pass


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
