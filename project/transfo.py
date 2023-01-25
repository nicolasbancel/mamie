from matplotlib import pyplot as plt
import numpy as np
from constant import *
from utils import *


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
        (2 * morph_size + 1, 2 * morph_size + 1),
        (morph_size, morph_size),
    )
    operation = morph_op_dic[morph_operator]
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
        show("Image processing", img_stack)

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
