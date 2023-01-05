import sys, os

project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_dir)
from transfo import *

#############################################
# JUST SETTING UP THE IMAGE
#############################################

file_name = "mamie0008.jpg"
original = load_original(file_name)
rectangles = whiten_edges(original)
borders = add_borders(rectangles, show_image=True)
grey = grey_original(borders, show_image=True)
ret, thresh = build_threshold(
    grey, constant.THRESH_MIN, constant.THESH_MAX, cv2.THRESH_BINARY_INV, True
)

#############################################
# TEST THRESHOLDS
#############################################


def multiple_transformations_tresholding_more(
    img_grey, file_name, THRESH_MIN=250, THESH_MAX=255
):
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
