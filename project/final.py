from transfo import *
from contour import *
from testing import *
from constant import *
from utils import *
from typing import Literal

#############################################
# [ISSUE] with white
# [FIXED] Works with 240-255
# PICTURE_NAME = "mamie0005.jpg"

# [ISSUE] with closeness
# [FIXED] Works with 240-255
# PICTURE_NAME = "mamie0008.jpg"

# [ISSUE] with white + closeness
# [FIXED] Works with 240-255
# PICTURE_NAME = "mamie0037.jpg"

# with config 240-255 : all good
# PICTURE_NAME = "mamie0017.jpg"
#############################################


def final_steps(picture_name, THRESH_MIN=240, THESH_MAX=255, export: Literal["all", "fail_only"] = "fail_only"):
    original = load_original(picture_name)
    original = whiten_edges(original)
    original = add_borders(original)
    img_grey = grey_original(original)
    img_blur = cv2.bilateralFilter(img_grey, 9, 75, 75)
    # img_blur = cv2.GaussianBlur(img_grey, (3, 3), 0)
    thresh = cv2.threshold(img_blur, THRESH_MIN, THESH_MAX, cv2.THRESH_BINARY_INV)[1]
    contours, _ = find_contours(source=thresh)
    original_with_main_contours, PictureContours, keyboard, message = draw_main_contours(
        original,
        contours,
        num_biggest_contours=6,
        contour_size=40,
        contours_color=(0, 255, 0),
        precision_param=0.01,
        only_rectangles=False,
        show_image=False,
    )

    # pdb.set_trace()
    message["picture_name"] = picture_name
    message["rm_black_edges"] = True
    message["add_white_margin"] = True
    message["blur_method"] = "bilateralFilter"
    message["blur_parameters"] = "(d, sigmaColor, sigmaSpace) = (9, 75, 75)"
    # message["blur_parameters"] = "kernelsize = (3, 3)"
    message["threshold"] = True
    message["threshold_method"] = "THRESH_BINARY_INV"
    message["threshold_min"] = THRESH_MIN
    message["threshold_max"] = THESH_MAX

    list_images = [original, img_blur, thresh, original_with_main_contours]
    list_labels = ["Original", "Blurred", "Threshold", "Original w contours"]

    # list_images = [original, img_grey, img_blur, thresh, original_with_main_contours]
    # list_labels = ["Original", "Grey", "Blurred", "Threshold", "Original w contours"]

    final = stack_images(list_labels, list_images, message, num_columns=4)
    # show("Final", final)

    PATH = "images/contouring/"

    if message["success"] == True:
        if export == "all":
            success_path = PATH + "success/" + picture_name
            cv2.imwrite(success_path, final)
    else:
        if export == "all" or "fail_only":
            failure_path = PATH + "failure/" + picture_name
            cv2.imwrite(failure_path, final)

    return message


if __name__ == "__main__":
    # THRESH_MIN = 252  # Identifies the biggest contour of the 2 pictures in 1 massive rectangle

    # THRESH_MIN = 240  # GOOD - TO KEEP [RUN #1]
    # THRESH_MIN = 252  # TESTING VS 250 TO HAVE ONLY THE PHOTOCOPIEUSE WHITE CONSIDERED WHITE

    # THRESH_MIN = 250 # TEST RUN #2

    THRESH_MIN = 245  # Test - RUN # 2
    THESH_MAX = 255  # GOOD - TO KEEP

    CONFIG_NUM = config_num()
    FINAL_MESSAGE = {
        "total_num_contours": [],
        "num_biggest_contours": [],
        "num_detected_photos_on_mosaic": [],
        "num_photos_on_mosaic": [],
        "photos_areas": [],
        "success": [],
        "picture_name": [],
        "rm_black_edges": [],
        "add_white_margin": [],
        "blur_method": [],
        "blur_parameters": [],
        "threshold": [],
        "threshold_method": [],
        "threshold_min": [],
        "threshold_max": [],
        "config_num": [],
    }

    for file in os.listdir(MOSAIC_DIR):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg") or filename.endswith(".png"):
            print(f"\n Extracting contours of file : {filename} \n")
            message = final_steps(filename, THRESH_MIN, THESH_MAX)
            for key in set(FINAL_MESSAGE) - {"config_num"}:
                FINAL_MESSAGE[key].append(message[key])
            FINAL_MESSAGE["config_num"].append(CONFIG_NUM)
        else:
            print("Else")
            continue

    log_results(FINAL_MESSAGE)
    # print(FINAL_MESSAGE)
