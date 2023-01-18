from transfo import *
from contour import *
from testing import *
from constant import *
from utils import *
from angles import *
from typing import Literal
import argparse
import pdb

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


THRESHOLD_NUM_POINTS_PER_CONTOUR = 6
# Because scission and because duplicate at the beginning and end


def final_steps(picture_name, THRESH_MIN, THESH_MAX, export: Literal["all", "fail_only"] = "fail_only"):
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

    final_contours, split_contours = fix_contours(PictureContours, original)
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

    list_images = [original, img_blur, thresh, original_with_main_contours, split_contours]
    list_labels = ["Original", "Blurred", "Threshold", "Original w contours", "Original w splitted contours"]

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

    return original, original_with_main_contours, PictureContours, final_contours, message


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=False, help="Name of image - located in mosaic dir")
    args = vars(ap.parse_args())

    # print(args["image"])
    # print(len(args))

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

    if args["image"] is not None:
        # Enables :
        # python3 final.py -i "mamie0001.jpg" to work and display only 1 image
        # python3 final.py -i "mamie0008.jpg"
        original, original_with_main_contours, PictureContours, final_contours, message = final_steps(args["image"], THRESH_MIN, THESH_MAX, export="all")
        # pdb.set_trace()
    else:
        "Iterate through all images + log in the results.csv file"
        # for file in os.listdir(MOSAIC_DIR)[6:12]:
        for file in sorted(os.listdir(MOSAIC_DIR)):
            filename = os.fsdecode(file)
            if filename.endswith(".jpg") or filename.endswith(".png"):
                print(f"\n Extracting contours of file : {filename} \n")
                original, original_with_main_contours, PictureContours, final_contours, message = final_steps(filename, THRESH_MIN, THESH_MAX, export="all")
                for key in set(FINAL_MESSAGE) - {"config_num"}:
                    FINAL_MESSAGE[key].append(message[key])
                FINAL_MESSAGE["config_num"].append(CONFIG_NUM)
            else:
                print("Else")
                continue

        log_results(FINAL_MESSAGE)
    # print(FINAL_MESSAGE)

    ### TESTING END TO END : TO BE COMMENTED OUT AND INDENTED
    """
    from transfo import *
    from contour import *
    from testing import *
    from constant import *
    from utils import *
    from angles import *
    from typing import Literal
    import argparse
    import pdb


    THRESH_MIN = 245
    # For mamie0047 : 252 works OK, 245 fucks it up
    THESH_MAX = 255

    THRESHOLD_NUM_POINTS_PER_CONTOUR = 6
    SMALL_ANGLE_THRESH = 7
    THRESHOLD = 0.25
    MAX_AREA_THRESHOLD = 10000000
    MIN_AREA_THRESHOLD = 6000000


    picture_name = "mamie0047.jpg"
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

    # Fix contour
    test = original.copy()
    final_image = original.copy()
    final_contours = []
    color_index = 0

    contour_info = PictureContours[0]
    contour = contour_info[0]
    contour_area = contour_info[1]
    if contour_area > MAX_AREA_THRESHOLD:
        cv2.drawContours(test, [contour], -1, (0, 255, 0), 40)
        # UNCOMMENT FOR TESTING
        # show("Contour", test)

        # GETTING ANGLES
        angles_degrees, alengths, blengths = get_angles(contour)
        # plot_angles(contour, angles_degrees)
        enriched_contour, scission_information, middle_point, scission_point, max_side_length = enrich_contour_info(contour, angles_degrees, alengths, blengths)
        # pdb.set_trace()
        if scission_point is not None:
            cv = plot_points(angles_degrees, enriched_contour, contour, middle_point)
            extrapolated_point = find_extrapolation(middle_point, scission_point, max_side_length)
            # new_contours, intersection_point = split_contour(contour, extrapolated_point, scission_point, middle_point, original, cv)
            new_contours, intersection_point = split_contour(contour, extrapolated_point, scission_point, middle_point, original)
        else:
            # new_contours has to be a list - in this case, it's a list of 1 single element
            new_contours = [contour]
        for cont in new_contours:
            # print(f"Contour is too big - color index = {color_index}")
            final_contours.append(cont)
            draw(final_image, cont, color_index)
            color_index += 1
    elif contour_area > MIN_AREA_THRESHOLD and contour_area <= MAX_AREA_THRESHOLD:
        # print(f"Contour has good shape - no need for split - color index = {color_index}")
        # Reduce size of contour
        contour = contour[:, 0, :]
        final_contours.append(contour)
        draw(final_image, contour, color_index)
        color_index += 1

    show("Final contours", final_image)
    # return final_contours, final_image
    """
