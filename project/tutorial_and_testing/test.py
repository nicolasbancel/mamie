import sys, os

project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_dir)
from transfo import *


#############################################
# Comments

# "mamie0008.jpg" :
# - BINARY_INV - 250 to 255 is the best one
# - Although it cannot really identify the white line in the middle

# "mamie0037.jpg" :
# - Problem is exactly the same as "mamie0008.jpg"
# - In addition to that, there's some black content within the bottom white rectangle
#   which end up being interpreted as a "hole" - hence the contour goes "inside" the picture

#
#############################################


#############################################
# JUST SETTING UP THE IMAGE
#############################################

# file_name = "mamie0008.jpg"
file_name = "mamie0037.jpg"
original = load_original(file_name)
rectangles = whiten_edges(original)
borders = add_borders(rectangles, show_image=True)
grey = grey_original(borders, show_image=True)
ret, thresh = build_threshold(grey, constant.THRESH_MIN, constant.THESH_MAX, cv2.THRESH_BINARY_INV, True)

#############################################
# FONCTION DEFINITIONS - THRESHOLDS
#############################################


def multiple_transformations_tresholding_more(img_grey, file_name, THRESH_MIN=250, THESH_MAX=255):
    # THRESH_MIN = 250
    # THESH_MAX = 255
    ret, thresh1 = cv2.threshold(img_grey, THRESH_MIN, THESH_MAX, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(img_grey, THRESH_MIN, THESH_MAX, cv2.THRESH_BINARY_INV)
    ret, thresh3 = cv2.threshold(img_grey, 127, THESH_MAX, cv2.THRESH_BINARY_INV)
    ret, thresh4 = cv2.threshold(img_grey, THRESH_MIN, THESH_MAX, cv2.THRESH_TRUNC)
    ret, thresh5 = cv2.threshold(img_grey, THRESH_MIN, THESH_MAX, cv2.THRESH_TOZERO)
    ret, thresh6 = cv2.threshold(img_grey, THRESH_MIN, THESH_MAX, cv2.THRESH_TOZERO_INV)

    ## Additions (for test)

    thresh7 = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh8 = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    ret, thresh9 = cv2.threshold(img_grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #  Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img_grey, (5, 5), 0)
    ret, thresh10 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    titles = [
        "Original Image - Grey",
        f"BINARY - {THRESH_MIN} to {THESH_MAX}",
        f"BINARY_INV - {THRESH_MIN} to {THESH_MAX}",
        f"BINARY_INV - 127 to {THESH_MAX}",
        "TRUNC",
        "TOZERO",
        "TOZERO_INV",
        "ADAPTIVE - MEAN",
        "ADAPTIVE - GAUSSIAN",
        "OTSU",
        "OTSU AFTER BLUR",
    ]
    images = [
        img_grey,
        thresh1,
        thresh2,
        thresh3,
        thresh4,
        thresh5,
        thresh6,
        thresh7,
        thresh8,
        thresh9,
        thresh10,
    ]

    cv2.imwrite(f"{file_name.split('.')[0]}_thresh_binary_inv.jpg", thresh2)

    # If n pictures to display :
    # - range(n-1)
    # - subplot(p, q) where p * q <= n

    for i in range(10):
        plt.subplot(4, 3, i + 1), plt.imshow(images[i], "gray", vmin=0, vmax=255)
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()


#############################################
# EXECUTION
#############################################

multiple_transformations_tresholding_more(grey, file_name, THRESH_MIN=250, THESH_MAX=255)
