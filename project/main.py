from transfo import *
from contour import *


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
    kernel = cv2.getStructuringElement(cv.MORPH_CROSS, (10, 10))
    erosion_two_iteration = cv2.erode(thresh, kernel, iterations=2)
    contours, _ = find_contours(source=erosion_two_iteration)
    original_with_main_contours, PictureContours = draw_main_contours(
        original_with_border,
        contours,
        num_contours=6,
        contour_size=40,
        contours_color=(0, 255, 0),
        only_rectangles=False,
        show_image=True,
    )


def iterate(dir):
    for file in os.listdir(mosaic_dir):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg") or filename.endswith(".png"):
            print(f"\n Extracting contours of file : {filename} \n")
            all_steps(filename)
        else:
            continue


if __name__ == "__main__":
    iterate(constant.MOSAIC_DIR)
