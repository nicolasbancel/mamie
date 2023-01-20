from constant import *
from pathlib import Path
import numpy as np
from utils import *
import pdb


def extract_contour(original, contour):
    # print(f"Printing contour # {idx + 1}")
    mask = np.zeros_like(original)
    # List of 1 element. Index -1 is for printing "all" elements of that list
    cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
    out = np.zeros_like(original)
    out[mask == 255] = original[mask == 255]
    # show("out", out)
    # np.where(mask == 255) results in a 3 dimensional array
    (y, x, z) = np.where(mask == 255)
    (topy, topx) = (np.min(y), np.min(x))
    (bottomy, bottomx) = (np.max(y), np.max(x))
    output_img = out[topy : bottomy + 1, topx : bottomx + 1]
    return output_img


def rotate_contour(original, contour):
    """
    Inspired from : https://theailearner.com/tag/opencv-rotation-angle/
    https://www.youtube.com/watch?v=SQ3D1tlCtNg&t=558s&ab_channel=GiovanniCode

    Loading the original is unnecessary
    """
    rectangle = cv2.minAreaRect(contour)
    (center, (width, height), angle) = rectangle
    rect_points = np.intp(cv2.boxPoints(rectangle))

    # Not used
    # rot_matrix = cv2.getRotationMatrix2D(center, angle, scale=1)

    ## Better understand boundaries of rectangle

    points = rect_points.reshape(4, 2)
    input_points = np.zeros((4, 2), dtype="float32")
    # input_points = np.zeros((4, 2), dtype="int64")
    points_sum = points.sum(axis=1)
    points_diff = np.diff(points, axis=1)

    # Top left and bottom right points : get by summing x and y coordinates
    # Top right and bottom left : difference between x and y coordinates

    # Top left : smallest sum : lowest sum : they are the closest from the
    # Bottom right : biggest sum (they are the furthest from the origin)

    # Top left corner
    input_points[0] = points[np.argmin(points_sum)]
    # Top right
    input_points[1] = points[np.argmin(points_diff)]

    # Bottom right
    input_points[3] = points[np.argmax(points_sum)]
    # Bottom left
    input_points[2] = points[np.argmax(points_diff)]

    # Switch 2 and 3 if want to draw contour in the right order
    # We use this order because of match with "converted points"

    # Un necessary part
    # corners = original.copy()
    # draw(corners, input_points, color_index=0, show_points=True, show_index=True, legend=["Top left", "Top right", "Bottom right", "Bottom left"])
    # show("o", corners)

    (top_left, top_right, bottom_right, bottom_left) = input_points

    bottom_width = np.sqrt((bottom_right[0] - bottom_left[0]) ** 2 + (bottom_right[1] - bottom_left[1]) ** 2)
    top_width = np.sqrt((top_right[0] - top_left[0]) ** 2 + (top_right[1] - top_left[1]) ** 2)

    right_height = np.sqrt((top_right[0] - bottom_right[0]) ** 2 + (top_right[1] - bottom_right[1]) ** 2)
    left_height = np.sqrt((top_left[0] - bottom_left[0]) ** 2 + (top_left[1] - bottom_left[1]) ** 2)

    max_width = max(int(bottom_width), int(top_width))
    max_height = max(int(right_height), int(left_height))

    converted_points = np.float32([[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]])

    matrix = cv2.getPerspectiveTransform(input_points, converted_points)
    output_img = cv2.warpPerspective(original, matrix, (max_width, max_height))

    # show("Output Perspective", img_output)

    """
    Test : using angled transformation
    
    # Testing boundingRectangle (from Stack Overflow)
    Y = max_height
    X = max_width

    x, y, w, h = cv2.boundingRect(rect_points)
    roi = original[y : y + h, x : x + w]

    # show("roi", roi)

    warp_rotate_dst_yx = cv2.warpAffine(roi, rot_matrix, (Y, X))
    show("Rotated", warp_rotate_dst_yx)

    warp_rotate_dst_xy = cv2.warpAffine(roi, rot_matrix, (X, Y))
    show("Rotated", warp_rotate_dst_xy)

    # pivoted = cv2.warpAffine(copy, rot_matrix, (to_rotate.shape, flags=cv2.INTER_LINEAR)
    pass
    """

    return output_img


def draw_rectangle_box(img, contour, rectangle):
    """
    The angle provided is the angle between first and last point of the contour
    So it's always a number between 0 and -90
    """
    copy = original.copy()
    rectangle = cv2.minAreaRect(contour)
    (center, (width, height), angle) = rectangle
    rect_points = np.intp(cv2.boxPoints(rectangle))

    x, y, w, h = cv2.boundingRect(rect_points)
    roi = copy[y : y + h, x : x + w]

    # Draw main contour
    draw(copy, contour, color_index=0, show_points=True, show_index=True)
    # Draw bounding rectangle contour
    draw(copy, rect_points, color_index=1, show_points=True, show_index=True)
    show("x", copy)
    return copy


def output(original, picture_name, contours: list, success: bool):
    """
    Takes 3 arguments :
    - the original image
    - the list of contours on that image
    - whether the contours are well suited
    """
    if success == True:

        for idx, contour in enumerate(contours):
            output_img = extract_contour(original, contour)
            # output_img = rotate_contour(original, contour)
            # in case picture_name is provided as a path
            # filename = Path(picture_name).stem
            # pdb.set_trace()
            (filename, extension) = picture_name.split(".")
            if idx + 1 < 10:
                suffix = "_0" + str(idx + 1)
            else:
                suffix = "_" + str(idx + 1)
            new_filename = filename + suffix + "." + extension
            path = os.path.join(CROPPED_DIR, new_filename)

            cv2.imwrite(path, output_img)


if __name__ == "__main__":
    picture_name = "mamie0001.jpg"

    # picture_name = "mamie0009.jpg"
    # contour_index = 2
    # Issue with picture mamie0009_03.jpg

    # in ipython :
    # from final import *
    # picture_name = "mamie0009.jpg"

    original, original_w_main_contours, original_w_final_contours, main_contours, final_contours, message = final_steps(
        picture_name, THRESH_MIN, THESH_MAX, export="all"
    )
    copy = original.copy()
    contour = final_contours[2]
    draw(copy, final_contours[2], show_points=True, show_index=True)

    success = message["success"] == True
    # export(original, picture_name, final_contours, success)

"""
## TEST END TO END
from final import *

picture_name = "mamie0009.jpg"
original, original_w_main_contours, original_w_final_contours, main_contours, final_contours, message = final_steps(
    picture_name, THRESH_MIN, THESH_MAX, export="all"
)
copy = original.copy()
contour = final_contours[2]
draw(copy, final_contours[2], show_points=True, show_index=True)
success = message["success"] == True
show("copy contour", copy)
"""
