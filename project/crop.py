from constant import *
from Mosaic import *
from Contour import *
from pathlib import Path
import numpy as np
from utils import *
import pdb
from Picture import *


def extract_contour(mosaic, contour):
    # Extracts the image delimited by a contour, from a mosaic
    # print(f"Printing contour # {idx + 1}")
    mask = np.zeros_like(mosaic.img_source)
    # List of 1 element. Index -1 is for printing "all" elements of that list
    cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
    out = np.zeros_like(mosaic.img_source)
    out[mask == 255] = mosaic.img_source[mask == 255]
    # show("out", out)
    # np.where(mask == 255) results in a 3 dimensional array
    (y, x, z) = np.where(mask == 255)
    (topy, topx) = (np.min(y), np.min(x))
    (bottomy, bottomx) = (np.max(y), np.max(x))
    output_img = out[topy : bottomy + 1, topx : bottomx + 1]
    return output_img


def euc_dist(point1: tuple, point2: tuple):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def warpAffine_contour(mosaic, contour, show_image=None):
    """
    A lot inspired from : https://github.com/sebastiengilbert73/tutorial_affine_perspective/blob/main/compute_transforms.py
    https://towardsdatascience.com/perspective-versus-affine-transformation-25033cef5766
    """
    orig_width, orig_height, _ = mosaic.img_source.shape
    warped = mosaic.img_source.copy()
    rectangle = cv2.minAreaRect(contour.points)
    (center, (width, height), angle) = rectangle
    width_int = int(width)
    height_int = int(height)
    feature_points_int = np.array(cv2.boxPoints(rectangle), dtype="int")
    feature_points = np.array(cv2.boxPoints(rectangle), dtype=np.float32)
    # Rearrange feature_points to be in the order of target points

    offset = 0  # or 100
    target_topleft = [offset, offset]
    target_topright = [offset + width_int, offset]
    target_bottomright = [offset + width_int, offset + height_int]
    target_bottomleft = [offset, offset + height_int]

    # Understand order in which rectangle points are ordered
    # target_topleft -> target_topright corresponds to ta width
    # target_topright -> target_bottomright corresponds to a height
    first_point = feature_points[0]
    second_point = feature_points[1]
    third_point = feature_points[2]
    fourth_point = feature_points[3]

    dist_01 = int(euc_dist(first_point, second_point))
    dist_12 = int(euc_dist(second_point, third_point))
    dist_23 = int(euc_dist(third_point, fourth_point))
    dist_30 = int(euc_dist(fourth_point, first_point))

    pixel_margin = 10

    # Make the target contour match the order of points of the source contour
    if width_int - pixel_margin <= dist_01 <= width_int + pixel_margin or width_int - pixel_margin <= dist_23 <= width_int + pixel_margin:
        warped_feature_points_int = np.array([target_topleft, target_topright, target_bottomright, target_bottomleft])
    elif height_int - pixel_margin <= dist_01 <= height_int + pixel_margin or height_int - pixel_margin <= dist_23 <= height_int + pixel_margin:
        warped_feature_points_int = np.array([target_topright, target_bottomright, target_bottomleft, target_topleft])
    else:
        warped_feature_points_int = None
        print(f"Width = {width_int} // Height = {height_int}\ndist_01 : {dist_01} - dist_12 : {dist_12}\ndist_23 : {dist_23} - dist_30 : {dist_30}")

    if show_image:
        copy = mosaic.img_source.copy()
        draw(copy, feature_points_int, color_index=0, show_points=True, show_index=True)
        draw(copy, warped_feature_points_int, color_index=0, show_points=True, show_index=True)
        show("WarpAffine - Transformed Img", copy)

    # After drown : 0 : Top left. 1 : Top right. 2 : Bottom right. 3 : Bottom left
    # warped_feature_points = np.array([[100, 100], [100 + width_int, 100], [100 + width_int, 100 + height_int], [100, 100 + height_int]], dtype=np.float32)
    # warped_feature_points_int = np.array([[0, 0], [0 + width_int, 0], [0 + width_int, 0 + height_int], [0, 0 + height_int]])

    warped_feature_points_float = np.array(warped_feature_points_int, dtype=np.float32)

    affine_mtx = cv2.getAffineTransform(feature_points[:3, :], warped_feature_points_float[:3, :])

    warped_image_size = (width_int, height_int)
    output_img = cv2.warpAffine(warped, affine_mtx, dsize=warped_image_size)

    if show_image:
        show("warpedAffine", output_img)

    return output_img


def warpPerspective_contour(mosaic, contour, show_image=None):
    """
    Inspired from : https://theailearner.com/tag/opencv-rotation-angle/
    https://www.youtube.com/watch?v=SQ3D1tlCtNg&t=558s&ab_channel=GiovanniCode

    Loading the original is unnecessary
    """
    original = mosaic.img_source
    rectangle = cv2.minAreaRect(contour.points)
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

    # input_points_drawing = input_points.astype(dtype="int64")
    # Un necessary part
    # corners = original.copy()
    # draw(corners, input_points_drawing, color_index=0, show_points=True, show_index=True, legend=["Top left", "Top right", "Bottom left", "Bottom right"])
    # If feeding "draw" with input_points (which are float) : getting "error: (-215:Assertion failed) npoints > 0 in function 'drawContours'" error
    # show("o", corners)

    # WATCH OUT WITH THIS ORDER : IT IS KEY
    # MAKE SURE IT'S CORRECT BY EXECUTING CODE ABOVE (DRAW)
    (top_left, top_right, bottom_left, bottom_right) = input_points

    bottom_width = np.sqrt((bottom_right[0] - bottom_left[0]) ** 2 + (bottom_right[1] - bottom_left[1]) ** 2)
    top_width = np.sqrt((top_right[0] - top_left[0]) ** 2 + (top_right[1] - top_left[1]) ** 2)

    right_height = np.sqrt((top_right[0] - bottom_right[0]) ** 2 + (top_right[1] - bottom_right[1]) ** 2)
    left_height = np.sqrt((top_left[0] - bottom_left[0]) ** 2 + (top_left[1] - bottom_left[1]) ** 2)

    max_width = max(int(bottom_width), int(top_width))
    max_height = max(int(right_height), int(left_height))

    converted_points = np.float32([[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]])

    matrix = cv2.getPerspectiveTransform(input_points, converted_points)
    output_img = cv2.warpPerspective(original, matrix, (max_width, max_height))

    if show_image:
        show("Output Perspective", output_img)

    return output_img


def draw_rectangle_box(original, contour):
    """
    The angle provided is the angle between first and last point of the contour
    So it's always a number between 0 and -90
    """
    copy = original.copy()
    rectangle = cv2.minAreaRect(contour)
    (center, (width, height), angle) = rectangle
    rect_points = np.intp(cv2.boxPoints(rectangle))

    x, y, w, h = cv2.boundingRect(rect_points)
    bounding_rect_contour = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])

    roi = copy[y : y + h, x : x + w]

    # Draw main contour
    draw(copy, contour, color_index=0, show_points=True, show_index=True)
    # Draw rectangle contour (which is supposed to be equal to the contour) : this one is rotated
    draw(copy, rect_points, color_index=1, show_points=True, show_index=False)
    # Draw bounding contour (this one is NOT rotated)
    draw(copy, bounding_rect_contour, color_index=1, show_points=True, show_index=False)
    show("x", copy)
    return copy


def crop(mosaic, export=None):
    """
    Takes the mosaic as an input (with its final contours) :
    - Does a warp affine transformation on each image within the contour
    - Adds the final images (warped) to the Mosaic attributes
    """
    cropped_images = dict({"filename": [], "img": []})
    for idx, contour in enumerate(mosaic.contours_final):
        # Option : before wrap
        # output_img = extract_contour(original, contour)
        output_img = warpAffine_contour(original, contour)
        # Option below looses quality
        # output_img = warpPerspective_contour(original, contour)
        (filename, extension) = mosaic.mosaic_name.split(".")
        if idx + 1 < 10:
            suffix = "_0" + str(idx + 1)
        else:
            suffix = "_" + str(idx + 1)
        new_filename = filename + suffix + "." + extension

        cropped_images["filename"].append(new_filename)
        cropped_images["img"].append(output_img)
        mosaic.cropped_images = cropped_images

        if export == True:
            path = os.path.join(CROPPED_DIR, new_filename)
            cv2.imwrite(path, output_img)

        print(f"Cropping done for image: {new_filename}")


if __name__ == "__main__":
    pass
