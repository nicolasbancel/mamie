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
import shutil

#############################################
# Image
# python3 final.py --image "mamie0008.jpg"
#############################################


EXECUTION_TIME = str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def get_contours(mosaic_name, export_contoured: Literal["all", "fail_only", "none"] = None, show_image=None):
    MAPPING_DICT = load_metadata(filename="pictures_per_mosaic.csv")
    mosaic = Mosaic(dir="source", mosaic_name=mosaic_name)
    find_contours(mosaic, retrieval_mode=cv2.RETR_EXTERNAL)  # updates mosaic.contours_all
    message = draw_main_contours(mosaic, only_rectangles=False, show_image=show_image)  # Updates 4 attributes :
    # mosaic.contours_main / img_w_main_contours / num_contours_total / num_contours_main
    fix_contours(mosaic, show_image=show_image)  # Updates 2 attributes : mosaic.contours_final & mosaic.img_w_final_contours
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
        # Pushing the failing file to the "TO TREAT" folder so that the cropping + rotation
        # is done manually there
        shutil.copy2(os.path.join(SOURCE_DIR, mosaic_name), TO_TREAT_DIR)
        if export_contoured == "all" or "fail_only":
            failure_path = CONTOURED_DIR + "failure/" + mosaic_name
            cv2.imwrite(failure_path, final)

    return mosaic, message


def all_steps(mosaic_name, export_contoured=None, export_cropped=None, export_rotated=None, show_contouring=None, show_cropping=None, show_rotation=None):
    # Get all contour information
    mosaic, log_contours = get_contours(mosaic_name, export_contoured=export_contoured, show_image=show_contouring)
    # Crop each contour, warpAffine it, and store the cropped images in a mosaic attribute
    log_dict = {
        "config_num": [],
        "picture_name": [],
        "rot90_true_num": [],
        "rot90_predicted_num": [],
        "success": [],
        "rot90_summary": [],
    }
    log_rot = log_dict.copy()  # to prevent local variable 'log_rot' referenced before assignment errors
    if mosaic.success == True:
        crop_mosaic(mosaic, export_cropped=export_cropped, show_image=show_cropping)
        # For each cropped Picture of the mosaic, get its correct rotation

        for i in range(mosaic.num_contours_final):
            picture_name = mosaic.cropped_pictures["filename"][i]
            cv2_array = mosaic.cropped_pictures["img"][i]
            picture = Picture(picture_name=picture_name, cv2_array=cv2_array)
            rotate_one(picture, export_rotated=export_rotated, show_steps=show_rotation)
            log_rot = fill_log(picture, EXECUTION_TIME, log_dict)
    return mosaic, log_contours, log_rot


def main(
    mosaic_list=None,
    num_mosaics=None,
    log_contouring=None,
    log_rotations=None,
    export_contoured=None,  # should be "all", or "fail_only"
    export_cropped=None,
    export_rotated=None,
    show_contouring=None,
    show_cropping=None,
    show_rotation=None,
):
    CONFIG_NUM = config_num()
    if mosaic_list is not None:
        mosaics_to_process = mosaic_list
    elif num_mosaics is not None:
        max_index = num_mosaics if num_mosaics < len(os.listdir(SOURCE_DIR)) else len(os.listdir(SOURCE_DIR)) - 1
        mosaics_to_process = sorted(os.listdir(SOURCE_DIR))[:max_index]
    else:
        mosaics_to_process = sorted(os.listdir(SOURCE_DIR))

    # Exception to be dealt with later (the scission point belongs to the polygon)
    to_remove = ["mamie0280.jpg", "mamie0124.jpg"]
    for elem in to_remove:
        if elem in mosaics_to_process:
            mosaics_to_process.remove(elem)

    for mosaic_name in mosaics_to_process:
        now = datetime.now()
        dt = now.strftime("%H:%M:%S")
        if mosaic_name.endswith(".jpg") or mosaic_name.endswith(".jpeg") or mosaic_name.endswith(".png"):
            print(f"\n Time is : {dt} - Treating : {mosaic_name} \n")
            mosaic, log_contours, log_rot = all_steps(
                mosaic_name, export_contoured, export_cropped, export_rotated, show_contouring, show_cropping, show_rotation
            )
            # print(f"Log rot : {log_rot}")
            for key in set(FINAL_LOG_CONTOURS) - {"config_num"}:
                FINAL_LOG_CONTOURS[key].append(log_contours[key])
            FINAL_LOG_CONTOURS["config_num"].append(CONFIG_NUM)
            for k in set(FINAL_LOG_ROTATIONS):
                # Need extend because after each mosaic run, we get a list of rotations x pictures
                FINAL_LOG_ROTATIONS[k].extend(log_rot[k])
        else:
            print(f"\n Time is : {dt} - Ignore - {mosaic_name} is not a picture file \n")
            continue
    if log_contouring == True:
        log_results(FINAL_LOG_CONTOURS, "results_contours.csv")
    if log_rotations == True:
        log_results(FINAL_LOG_ROTATIONS, "results_rotations.csv")


if __name__ == "__main__":

    ################################################################
    # FINAL RUNNING SCRIPT
    ################################################################
    # If exporting only failed contours
    # python3 main.py -log_c -log_r -exco "fail_only" -excr -exro --no-show_contouring --no-show_cropping --no-show_rotation

    # If exporting success and failures
    # python3 main.py -log_c -log_r -exco "all" -excr -exro --no-show_contouring --no-show_cropping --no-show_rotation

    ################################################################
    # OTHER TYPES OF SCRIPTS
    ################################################################

    # Show everything
    # python3 main.py -n 4 -log_c -exco "fail_only" -excr -exro -shco -shcr -shro

    # Show contouring, do not show cropping and rotation steps. Export data
    # python3 main.py -n 2 -log_c -exco "fail_only" -excr -exro -shco --no-show_cropping --no-show_rotation

    # Test with long list
    # python3 main.py -m "mamie0003.jpg" "mamie0000.jpg" "mamie0001.jpg" -log_c -exco "fail_only" -excr -exro --no-show_contouring --no-show_cropping --no-show_rotation
    # python3 main.py -m "mamie0022.jpg" --no-log_contouring --no-log_rotations -exco "all" -excr -exro --show_contouring --show_cropping --no-show_rotation

    # Test Lara
    # python3 main.py -m "lara0001.jpg" "lara0002.jpg" "lara0003.jpg" "lara0004.jpg" -exco "all" -excr -exro --show_contouring --show_cropping --show_rotation

    # python3 main.py -m "mamie0193.jpg" --no-log_contouring --no-log_rotations -exco "all" -excr -exro --show_contouring --show_cropping --no-show_rotation

    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--mosaic_list", nargs="+", required=False, help="Name of mosaic - located in source dir")
    ap.add_argument("-n", "--num_mosaics", required=False, type=int, help="Number of mosaics to process")
    ap.add_argument("-log_c", "--log_contouring", action=argparse.BooleanOptionalAction, help="Whether or not results should be logged in results_contours.csv")
    ap.add_argument("-log_r", "--log_rotations", action=argparse.BooleanOptionalAction, help="Whether or not results should be logged in results_rotations.csv")
    ap.add_argument(
        "-exco", "--export_contoured", required=False, choices=["all", "fail_only", "none"], help="Whether the script should export the contoured .jpg"
    )
    ap.add_argument("-excr", "--export_cropped", action=argparse.BooleanOptionalAction, help="Whether the script should export the cropped pictures")
    ap.add_argument("-exro", "--export_rotated", action=argparse.BooleanOptionalAction, help="Whether the script should export the rotated pictures")
    ap.add_argument("-shco", "--show_contouring", action=argparse.BooleanOptionalAction, help="Whether the script should show images of steps for contouring")
    ap.add_argument("-shcr", "--show_cropping", action=argparse.BooleanOptionalAction, help="Whether the script should show images of steps for cropping")
    ap.add_argument("-shro", "--show_rotation", action=argparse.BooleanOptionalAction, help="Whether the script should show images of steps for rotating")
    args = vars(ap.parse_args())

    mosaic_list = args["mosaic_list"]
    num_mosaics = args["num_mosaics"]
    log_contouring = args["log_contouring"]
    log_rotations = args["log_rotations"]
    export_contoured = args["export_contoured"]
    export_cropped = args["export_cropped"]
    export_rotated = args["export_rotated"]
    show_contouring = args["show_contouring"]
    show_cropping = args["show_cropping"]
    show_rotation = args["show_rotation"]

    print(
        f"mosaic_list : {mosaic_list} \
// num_mosaics : {num_mosaics} \
// log_contouring : {log_contouring} \
// log_rotations : {log_rotations} \
// export_contoured : {export_contoured} \
// export_cropped : {export_cropped} \
// export_rotated : {export_rotated} \
// show_contouring : {show_contouring} \
// show_cropping : {show_cropping} \
// show_rotation: {show_rotation}"
    )

    main(
        mosaic_list, num_mosaics, log_contouring, log_rotations, export_contoured, export_cropped, export_rotated, show_contouring, show_cropping, show_rotation
    )

    # Fixing Pt 1 and 2 too close to each other
    # python3 main.py -m "mamie0301.jpg" "mamie0302.jpg" "mamie0303.jpg" "mamie0304.jpg" "mamie0305.jpg" "mamie0306.jpg" --no-log_contouring --no-log_rotations -exco "all" -excr -exro --show_contouring --show_cropping --no-show_rotation
