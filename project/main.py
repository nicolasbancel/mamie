from constant import *
from Mosaic import *
from Contour import *

from Picture import *
from utils import *
from utils_contour import *
from typing import Literal
from crop import *
from rotate import *
import argparse
import pdb
from datetime import datetime

#############################################
# Image
# python3 final.py --image "mamie0008.jpg"
#############################################


def get_contours(mosaic_name, export_contoured: Literal["all", "fail_only", "none"] = None):
    MAPPING_DICT = load_metadata(filename="pictures_per_mosaic.csv")
    mosaic = Mosaic(mosaic_name)
    find_contours(mosaic, retrieval_mode=cv2.RETR_EXTERNAL)  # updates mosaic.contours_all
    message = draw_main_contours(mosaic, only_rectangles=False, show_image=True)  # Updates 4 attributes :
    # mosaic.contours_main / img_w_main_contours / num_contours_total / num_contours_main
    fix_contours(mosaic)  # Updates 2 attributes : mosaic.contours_final & mosaic.img_w_final_contours
    mosaic.num_points_per_contour = [len(cont) for cont in mosaic.contours_final]
    # inc represents the number of contours that have more than 6 points
    # which makes them wrong
    inc = 0
    for num_points in mosaic.num_points_per_contour:
        if num_points > THRESHOLD_NUM_POINTS_PER_CONTOUR:
            inc += 1

    if mosaic.num_contours_final == mosaic.true_num_pictures and inc == 0:
        mosaic.success = True
        message["success"] = True
    else:
        mosaic.success = False
        message["success"] = False

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

    message["execution_time"] = dt_string
    message["picture_name"] = mosaic_name
    message["rm_black_edges"] = True
    message["add_white_margin"] = True
    message["blur_method"] = "GaussianBlur"
    message["blur_parameters"] = "kernelsize = (3, 3)"
    message["threshold"] = True
    message["threshold_method"] = "THRESH_BINARY_INV"
    message["threshold_min"] = THRESH_MIN
    message["threshold_max"] = THESH_MAX
    message["split_contours"] = True

    message["true_num_pictures"] = mosaic.true_num_pictures
    message["num_contours_after_split"] = mosaic.num_contours_final
    message["num_points_per_contour"] = mosaic.num_points_per_contour

    list_images = [mosaic.img_source, mosaic.img_blur, mosaic.img_thresh, mosaic.img_w_main_contours, mosaic.img_w_final_contours]
    list_labels = ["Original", "Blurred", "Threshold", "Original w contours", "Original w final contours"]

    final = stack_images(list_labels, list_images, message, num_columns=4)
    # show("Final", final)

    if mosaic.success == True:
        if export_contoured == "all":
            success_path = CONTOURED_DIR + "success/" + mosaic_name
            cv2.imwrite(success_path, final)
    else:
        if export_contoured == "all" or "fail_only":
            failure_path = CONTOURED_DIR + "failure/" + mosaic_name
            cv2.imwrite(failure_path, final)

    return mosaic, message


def all_steps(mosaic_name, export_contoured="fail_only", export_cropped="all", export_rotated="all"):
    # Get all contour information
    mosaic, message = get_contours(mosaic_name, export_contoured)
    # Crop each contour, warpAffine it, and store the cropped images in a mosaic attribute
    if mosaic.success == True:
        crop_mosaic(mosaic, export_cropped)
        # For each cropped Picture of the mosaic, get its correct rotation
        for i in range(mosaic.num_contours_final):
            picture_name = mosaic.cropped_pictures["filename"][i]
            cv2_array = mosaic.cropped_pictures["img"][i]
            picture = Picture(picture_name, cv2_array)
            rotate_one(picture, export_rotated=True, show_steps=True)


if __name__ == "__main__":
    all_steps("mamie0028.jpg")

    # Example of execution plan
    # python3 final.py -n 20 OR
    # python3 final.py OR
    # python3 final.py -i "mamie0001.jpg" OR
    # python3 final.py -log True

    """

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=False, help="Name of image - located in mosaic dir")
    ap.add_argument("-n", "--num_mosaics", required=False, type=int, help="Number of mosaics to process")
    ap.add_argument("-log", "--log_results", required=False, nargs="?", const=False, help="Whether or not results should be logged in results.csv")
    args = vars(ap.parse_args())

    # print(args["image"])
    # print(len(args))

    CONFIG_NUM = config_num()
    FINAL_MESSAGE = {
        "execution_time": [],
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
        # for file in os.listdir(SOURCE_DIR)[6:12]:
        if args["num_mosaics"] is None:
            # print(f"Will process all images")

            # mosaics_to_process = sorted(os.listdir(SOURCE_DIR))
            mosaics_to_process = sorted(os.listdir(SOURCE_DIR))[67:]
        else:
            print(f"Will process images from index 0 until {args['num_mosaics'] - 1}")
            mosaics_to_process = sorted(os.listdir(SOURCE_DIR))[: args["num_mosaics"]]
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
            log_results(FINAL_MESSAGE, "results.csv")
    """
