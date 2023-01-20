from transfo import *
from contour import *
from constant import *
from utils import *
from angles import *
from typing import Literal
from export import *
import argparse
import pdb
from datetime import datetime

#############################################
# Image
# python3 final.py --image "mamie0008.jpg"
#############################################


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


def final_steps(picture_name, THRESH_MIN, THESH_MAX, export: Literal["all", "fail_only", "none"] = "fail_only"):
    MAPPING_DICT = num_pictures_per_mosaic(filename="pictures_per_mosaic.csv")
    true_num_pictures = int(MAPPING_DICT[picture_name])
    original = load_original(picture_name)
    original = whiten_edges(original)
    original = add_borders(original)
    img_grey = grey_original(original)
    # img_blur = cv2.bilateralFilter(img_grey, 9, 75, 75) # PAS EFFICACE
    img_blur = cv2.GaussianBlur(img_grey, (3, 3), 0)
    thresh = cv2.threshold(img_blur, THRESH_MIN, THESH_MAX, cv2.THRESH_BINARY_INV)[1]
    contours, _ = find_contours(source=thresh)
    original_w_main_contours, main_contours, keyboard, message = draw_main_contours(
        original,
        contours,
        contour_size=40,
        contours_color=(0, 255, 0),
        precision_param=0.01,
        only_rectangles=False,
        show_image=False,
    )

    final_contours, original_w_final_contours = fix_contours(main_contours, original)
    num_contours = len(final_contours)
    num_points_per_contour = [len(cont) for cont in final_contours]
    # inc represents the number of contours that have more than 6 points
    # which makes them wrong
    inc = 0
    for num_points in num_points_per_contour:
        if num_points > THRESHOLD_NUM_POINTS_PER_CONTOUR:
            inc += 1

    if num_contours == true_num_pictures and inc == 0:
        message["success"] = True
    else:
        message["success"] = False

    success = message["success"]

    # pdb.set_trace()
    message["picture_name"] = picture_name
    message["rm_black_edges"] = True
    message["add_white_margin"] = True
    message["blur_method"] = "GaussianBlur"
    # message["blur_method"] = "bilateralFilter"
    # message["blur_parameters"] = "(d, sigmaColor, sigmaSpace) = (9, 75, 75)"
    message["blur_parameters"] = "kernelsize = (3, 3)"
    message["threshold"] = True
    message["threshold_method"] = "THRESH_BINARY_INV"
    message["threshold_min"] = THRESH_MIN
    message["threshold_max"] = THESH_MAX
    message["split_contours"] = True

    message["true_num_pictures"] = true_num_pictures
    message["num_contours_after_split"] = num_contours
    message["num_points_per_contour"] = num_points_per_contour

    list_images = [original, img_blur, thresh, original_w_main_contours, original_w_final_contours]
    list_labels = ["Original", "Blurred", "Threshold", "Original w contours", "Original w final contours"]

    # list_images = [original, img_grey, img_blur, thresh, original_with_main_contours]
    # list_labels = ["Original", "Grey", "Blurred", "Threshold", "Original w contours"]

    final = stack_images(list_labels, list_images, message, num_columns=4)
    # show("Final", final)

    if success == True:
        if export == "all":
            success_path = CONTOURED_DIR + "success/" + picture_name
            # print(success_path)
            cv2.imwrite(success_path, final)
    else:
        if export == "all" or "fail_only":
            failure_path = CONTOURED_DIR + "failure/" + picture_name
            # print(failure_path)
            cv2.imwrite(failure_path, final)

    return original, original_w_main_contours, original_w_final_contours, main_contours, final_contours, message


if __name__ == "__main__":

    # Example of execution plan
    # python3 final.py -n 20 OR
    # python3 final.py OR
    # python3 final.py -i "mamie0001.jpg"

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=False, help="Name of image - located in mosaic dir")
    ap.add_argument("-n", "--num_mosaics", required=False, type=int, help="Number of mosaics to process")
    ap.add_argument("-log", "--log_results", required=False, nargs="?", const=False, help="Whether or not results should be logged in results.csv")
    args = vars(ap.parse_args())

    # print(args["image"])
    # print(len(args))

    CONFIG_NUM = config_num()
    FINAL_MESSAGE = {
        "total_num_contours": [],
        "num_biggest_contours": [],
        "num_rectangles_before_split": [],
        # "num_detected_photos_on_mosaic": [],
        # "num_photos_on_mosaic": [],
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
        "split_contours": [],
        "true_num_pictures": [],
        "num_contours_after_split": [],
        "num_points_per_contour": [],
        "config_num": [],
    }

    picture_name = args["image"]

    if picture_name is not None:
        # Enables :
        # python3 final.py -i "mamie0001.jpg" to work and display only 1 image
        # python3 final.py -i "mamie0008.jpg"
        original, original_w_main_contours, original_w_final_contours, main_contours, final_contours, message = final_steps(
            picture_name, THRESH_MIN, THESH_MAX, export="all"
        )
        success = message["success"]
        output(original, picture_name, final_contours, success)
        # pdb.set_trace()
    else:
        "Iterate through all images + log in the results.csv file"
        # for file in os.listdir(MOSAIC_DIR)[6:12]:
        if args["num_mosaics"] is None:
            # print(f"Will process all images")
            mosaics_to_process = sorted(os.listdir(MOSAIC_DIR))
        else:
            print(f"Will process images from index 0 until {args['num_mosaics'] - 1}")
            mosaics_to_process = sorted(os.listdir(MOSAIC_DIR))[: args["num_mosaics"]]
        for file in mosaics_to_process:
            now = datetime.now()
            dt = now.strftime("%H:%M:%S")
            filename = os.fsdecode(file)
            if filename.endswith(".jpg") or filename.endswith(".png"):
                print(f"\n Time is : {dt} - Extracting contours of file : {filename} \n")
                original, original_w_main_contours, original_w_final_contours, main_contours, final_contours, message = final_steps(
                    filename, THRESH_MIN, THESH_MAX, export="all"
                )
                success = message["success"]

                output(original, filename, final_contours, success)

                for key in set(FINAL_MESSAGE) - {"config_num"}:
                    FINAL_MESSAGE[key].append(message[key])
                FINAL_MESSAGE["config_num"].append(CONFIG_NUM)
            else:
                print(f"\n Time is : {dt} - Ignore - {filename} is not a picture file \n")
                continue
        # print(args["log_results"]) is None when not provided
        if args["log_results"] == True:
            log_results(FINAL_MESSAGE)
