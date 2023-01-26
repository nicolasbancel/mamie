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

FINAL_MESSAGE = {
    "execution_time": [],
    "total_num_contours": [],
    "num_biggest_contours": [],
    "num_rectangles_before_split": [],
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


def get_contours(mosaic_name, export_contoured: Literal["all", "fail_only", "none"] = None, show_image=None):
    MAPPING_DICT = load_metadata(filename="pictures_per_mosaic.csv")
    mosaic = Mosaic(mosaic_name)
    find_contours(mosaic, retrieval_mode=cv2.RETR_EXTERNAL)  # updates mosaic.contours_all
    message = draw_main_contours(mosaic, only_rectangles=False, show_image=show_image)  # Updates 4 attributes :
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


def all_steps(mosaic_name, export_contoured=None, export_cropped=None, export_rotated=None, show_contouring=None, show_cropping=None, show_rotation=None):
    # Get all contour information
    mosaic, message = get_contours(mosaic_name, export_contoured=export_contoured, show_image=show_contouring)
    # Crop each contour, warpAffine it, and store the cropped images in a mosaic attribute
    if mosaic.success == True:
        crop_mosaic(mosaic, export_cropped=export_cropped, show_image=show_cropping)
        # For each cropped Picture of the mosaic, get its correct rotation
        for i in range(mosaic.num_contours_final):
            picture_name = mosaic.cropped_pictures["filename"][i]
            cv2_array = mosaic.cropped_pictures["img"][i]
            picture = Picture(picture_name=picture_name, cv2_array=cv2_array)
            rotate_one(picture, export_rotated=export_rotated, show_steps=show_rotation)


def main(
    mosaic_list=None,
    num_mosaics=None,
    log_results=None,
    export_contoured="all",
    export_cropped=True,
    export_rotated=True,
    show_contouring=None,
    show_cropping=None,
    show_rotation=None,
):
    CONFIG_NUM = config_num()
    if mosaic_list is not None:
        # Enables :
        # python3 final.py -m ["mamie0001.jpg"] to work and treat only 1 mosaic
        # python3 final.py -m ["mamie0008.jpg"]
        for mosaic_name in mosaic_list:
            all_steps(mosaic_name, export_contoured, export_cropped, export_rotated, show_contouring, show_cropping, show_rotation)
    else:
        if num_mosaics is None:
            # Processing all mosaics
            mosaics_to_process = sorted(os.listdir(SOURCE_DIR))
        else:
            max_index = num_mosaics if num_mosaics < len(os.listdir(SOURCE_DIR)) else len(os.listdir(SOURCE_DIR)) - 1
            mosaics_to_process = sorted(os.listdir(SOURCE_DIR))[:max_index]

        for mosaic in mosaics_to_process:
            now = datetime.now()
            dt = now.strftime("%H:%M:%S")
            if mosaic.endswith(".jpg") or mosaic.endswith(".png"):
                print(f"\n Time is : {dt} - Treating : {mosaic} \n")
                all_steps(mosaic_name, export_contoured, export_cropped, export_rotated, show_contouring, show_cropping, show_rotation)
                for key in set(FINAL_MESSAGE) - {"config_num"}:
                    FINAL_MESSAGE[key].append(message[key])
                FINAL_MESSAGE["config_num"].append(CONFIG_NUM)
            else:
                print(f"\n Time is : {dt} - Ignore - {filename} is not a picture file \n")
                continue
    if log_results == True:
        log_results(FINAL_MESSAGE, "results.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--mosaic_list", required=False, help="Name of mosaic - located in source dir")
    ap.add_argument("-n", "--num_mosaics", required=False, type=int, help="Number of mosaics to process")
    ap.add_argument("-log", "--log_results", required=True, nargs="?", const=False, help="Whether or not results should be logged in results.csv")
    ap.add_argument(
        "-exco", "--export_contoured", required=False, choices=["all", "fail_only", "none"], help="Whether the script should export the contoured .jpg"
    )
    ap.add_argument("-excr", "--export_cropped", required=False, nargs="?", const=False, help="Whether the script should export the cropped pictures")
    ap.add_argument("-exro", "--export_rotated", required=False, nargs="?", const=False, help="Whether the script should export the rotated pictures")
    ap.add_argument("-shco", "--show_contouring", required=False, nargs="?", const=False, help="Whether the script should show images of steps for contouring")
    ap.add_argument("-shcr", "--show_cropping", required=False, nargs="?", const=False, help="Whether the script should show images of steps for cropping")
    ap.add_argument("-shco", "--show_rotation", required=False, nargs="?", const=False, help="Whether the script should show images of steps for rotating")
    args = vars(ap.parse_args())

    mosaic_list = args["mosaic_list"]
    num_mosaics = args["num_mosaics"]
    log_results = args["log_results"]
    export_contoured = args["export_contoured"]
    export_cropped = args["export_cropped"]
    export_rotated = args["export_rotated"]
    show_contouring = args["show_contouring"]
    show_cropping = args["show_cropping"]
    show_rotation = args["show_rotation"]

    #print(f"mosaic_list : {mosaic_list} // num_mosaics : {num_mosaics} // log_results : {log_results}")
    
    main(
        mosaic_list,
        num_mosaics
        log_results,
        export_contoured,
        export_cropped,
        export_rotated,
        show_contouring,
        show_cropping,
        show_rotation,
    )

    # all_steps("mamie0028.jpg")

    # Example of execution plan
    # python3 final.py -n 20 OR
    # python3 final.py OR
    # python3 final.py -i "mamie0001.jpg" OR
    # python3 final.py -log True
