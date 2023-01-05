from transfo import *
from contour import *
import argparse


def all_steps(picture_name):

    # original, img_grey = gray_image("mamie0001.jpg")

    original = load_original(picture_name)

    original_rectangles = whiten_edges(
        original, 15, 25, color=(255, 255, 255), show_image=False
    )
    original_with_border = add_borders(
        original_rectangles, color=(255, 255, 255), show_image=False
    )
    img_grey = grey_original(original_with_border)
    ret, thresh = build_threshold(
        img_grey, constant.THRESH_MIN, constant.THESH_MAX, cv2.THRESH_BINARY_INV
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (10, 10))
    erosion_two_iteration = cv2.erode(thresh, kernel, iterations=2)
    contours, _ = find_contours(source=erosion_two_iteration)
    original_with_main_contours, PictureContours, key = draw_main_contours(
        original_with_border,
        contours,
        num_contours=6,
        contour_size=40,
        contours_color=(0, 255, 0),
        only_rectangles=False,
        show_image=True,
    )

    return key


def iterate(dir):
    for file in os.listdir(dir):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg") or filename.endswith(".png"):
            print(f"\n Extracting contours of file : {filename} \n")
            key = all_steps(filename)
            if key == ord("q"):
                break
        else:
            continue


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-i", "--image", required=False, help="Name of image - located in mosaic dir"
    )
    args = vars(ap.parse_args())

    if len(args) == 1:
        # Enables : python3 main.py -i "mamie0001.jpg" to work and display only 1 image
        all_steps(args["image"])
    else:
        iterate(constant.MOSAIC_DIR)
    # print(args)
    # print(len(args))
    # print(ap.parse_args())
    # iterate(constant.MOSAIC_DIR)
