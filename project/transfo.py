import os
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import argparse
import cv2 as cv2
from constant import *

# parser = argparse.ArgumentParser(description="Code for Canny Edge Detector tutorial.")
# parser.add_argument("--input", help="Path to input image.", default="mamie0037.jpg")
# args = parser.parse_args()

# def load_original(file_name=args.input):


def load_original(file_name, dir="source"):
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


def whiten_edges(
    source,
    thickness_vertical=15,
    thickness_horizontal=25,
    color=(255, 255, 255),  # color=(0, 255, 0) for GREEN
    show_image=False,
):
    source_with_rectangles = source.copy()
    num_row, num_col = source.shape[:2]
    # thickness = 60
    top_left = (0, 0)
    bottom_right_vertical = (thickness_vertical, num_row)
    bottom_right_horizontal = (num_col, thickness_horizontal)

    # Adding horizontal rectangle on top
    cv2.rectangle(source_with_rectangles, top_left, bottom_right_vertical, color, -1)

    # Adding vertical rectangle on the left
    cv2.rectangle(source_with_rectangles, top_left, bottom_right_horizontal, color, -1)

    if show_image:
        cv2.imshow("With rectangles", source_with_rectangles)
        cv2.waitKey()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    return source_with_rectangles


def add_borders(source, color=(255, 255, 255), show_image=False):
    # Inspired from https://docs.opencv.org/3.4/dc/da3/tutorial_copyMakeBorder.html
    border_size = min(int(0.05 * source.shape[0]), int(0.05 * source.shape[1]))
    top = bottom = left = right = border_size
    borderType = cv2.BORDER_CONSTANT
    window_name = "Added blank border"
    value = color
    src = source.copy()

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    # 5% of width or height
    # Apply the same border to all borders (the min of the 2 above)

    # border_size = min(int(0.05 * original.shape[0]),int(0.05 * original.shape[1]))

    # original_with_border = cv.copyMakeBorder(src=src,top=border_size,bottom=border_size,left=border_size,right=border_size,borderType=borderType,None,value=color)
    original_with_border = cv2.copyMakeBorder(src, top, bottom, left, right, borderType, None, value)

    if show_image:
        cv2.imshow(window_name, original_with_border)
        cv2.waitKey()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    return original_with_border


def grey_original(source, show_image=False):
    img_grey = source.copy()
    img_grey = cv2.cvtColor(img_grey, cv2.COLOR_BGR2GRAY)
    if show_image:
        cv2.imshow("Grey image", img_grey)
        cv2.waitKey()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    return img_grey


def show(title, image):
    cv2.imshow(title, image)
    cv2.waitKey()
    cv2.destroyAllWindows()
    cv2.waitKey(1)


"""
Deprecated - this is confusing (and not bringing much value)
def write(filename, image, folder="processing"):
    # writing in subfolder images
    full_folder = "images/" + folder + "/"
    full_path = full_folder + filename
    print(full_path)
    cv2.imwrite(full_path, image)
"""


def multiple_transformations_tresholding(img_grey, file_name, THRESH_MIN=250, THESH_MAX=250):
    # THRESH_MIN = 250
    # THESH_MAX = 255
    ret, thresh1 = cv2.threshold(img_grey, THRESH_MIN, THESH_MAX, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(img_grey, THRESH_MIN, THESH_MAX, cv2.THRESH_BINARY_INV)
    ret, thresh3 = cv2.threshold(img_grey, THRESH_MIN, THESH_MAX, cv2.THRESH_TRUNC)
    ret, thresh4 = cv2.threshold(img_grey, THRESH_MIN, THESH_MAX, cv2.THRESH_TOZERO)
    ret, thresh5 = cv2.threshold(img_grey, THRESH_MIN, THESH_MAX, cv2.THRESH_TOZERO_INV)

    titles = [
        "Original Image - Grey",
        "BINARY",
        "BINARY_INV",
        "TRUNC",
        "TOZERO",
        "TOZERO_INV",
    ]
    images = [img_grey, thresh1, thresh2, thresh3, thresh4, thresh5]

    cv2.imwrite(f"{file_name.split('.')[0]}_thresh_binary_inv.jpg", thresh2)

    for i in range(6):
        plt.subplot(2, 3, i + 1), plt.imshow(images[i], "gray", vmin=0, vmax=255)
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()


def build_threshold(source, THRESH_MIN, THESH_MAX, method=cv2.THRESH_BINARY_INV, show_image=False):
    ret, thresh = cv2.threshold(source, THRESH_MIN, THESH_MAX, method)
    if show_image:
        show("After thresholding", thresh)
    return ret, thresh


def iterate_image_processing(source, thresh, morph_operator, morph_elem, show_image=False):

    element = cv2.getStructuringElement(
        morph_elem,
        (2 * constant.morph_size + 1, 2 * constant.morph_size + 1),
        (constant.morph_size, constant.morph_size),
    )
    operation = constant.morph_op_dic[morph_operator]
    morphex = cv2.morphologyEx(thresh, operation, element)

    edged = cv2.Canny(morphex, 30, 150, 3)
    dilated = cv2.dilate(edged, (1, 1), iterations=2)

    # This step is needed - otherwise the labelling can only be done
    # In Black or white, and it's hard to see
    thresh_multidim = np.stack((thresh,) * 3, axis=-1)
    morphex_multidim = np.stack((morphex,) * 3, axis=-1)
    edged_multidim = np.stack((edged,) * 3, axis=-1)
    dilated_multidim = np.stack((dilated,) * 3, axis=-1)

    images = [thresh, morphex, edged, dilated]
    images_multidim = [
        thresh_multidim,
        morphex_multidim,
        edged_multidim,
        dilated_multidim,
    ]
    images_names = ["Thresh", "Morph Ex", "Canny Edged", "Dilated"]
    img_stack = np.hstack(images_multidim)

    width = source.shape[1]
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 255, 0)  # green
    # color = (205, 0, 255) #
    # color = (255, 255, 255)

    for index, image_name in enumerate(images_names):
        image = cv2.putText(
            img_stack,
            f"{index + 1} - {image_name}",
            (5 + width * index, 500),
            font,
            8,  # fontScale (8 is fairly good)
            color,
            4,  # thickness
            cv2.LINE_AA,
        )
        # print(f'Text located at {5 + width * index} x 30')

    if show_image:
        cv2.imshow("Image processing", img_stack)
        cv2.waitKey()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    return images, img_stack


def label_stack(img_stacked, img_original, img_names: list, blackwhite: bool = True):
    if blackwhite:
        color = 0
    else:
        color = (0, 255, 0)  # green
    width = img_original.shape[1]
    font = cv2.FONT_HERSHEY_SIMPLEX
    for index, image_name in enumerate(img_names):
        image = cv2.putText(
            img_stacked,
            f"{index + 1} - {image_name}",
            (5 + width * index, 500),
            font,
            8,  # fontScale (8 is fairly good)
            color,
            # 4,  # thickness
            12,  # thickness
            cv2.LINE_AA,
        )
    return img_stacked
