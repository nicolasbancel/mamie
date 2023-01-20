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


def euc_dist(point1: tuple, point2: tuple):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def warpAffine_contour(original, contour, show_image=False):
    """
    A lot inspired from : https://github.com/sebastiengilbert73/tutorial_affine_perspective/blob/main/compute_transforms.py
    https://towardsdatascience.com/perspective-versus-affine-transformation-25033cef5766
    """
    orig_width, orig_height, _ = original.shape
    warped = original.copy()
    rectangle = cv2.minAreaRect(contour)
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

    # Make the target contour match the order of points of the source contour
    if dist_01 == width_int or dist_23 == width_int:
        warped_feature_points_int = np.array([target_topleft, target_topright, target_bottomright, target_bottomleft])
    elif dist_01 == height_int or dist_23 == height_int:
        warped_feature_points_int = np.array([target_topright, target_bottomright, target_bottomleft, target_topleft])
    else:
        warped_feature_points_int = None
        print(f"Width = {width_int} // Height = {height_int}\ndist_12 : {dist_12} - dist_34 : {dist_34}\ndist_23 : {dist_23} - dist_41 : {dist_41}")

    if show_image:
        copy = original.copy()
        draw(copy, feature_points_int, color_index=0, show_points=True, show_index=True)
        draw(copy, warped_feature_points_int, color_index=0, show_points=True, show_index=True)
        show("Copy with points", copy)

    # After drown : 0 : Top left. 1 : Top right. 2 : Bottom right. 3 : Bottom left
    # warped_feature_points = np.array([[100, 100], [100 + width_int, 100], [100 + width_int, 100 + height_int], [100, 100 + height_int]], dtype=np.float32)
    # warped_feature_points_int = np.array([[0, 0], [0 + width_int, 0], [0 + width_int, 0 + height_int], [0, 0 + height_int]])

    warped_feature_points_float = np.array(warped_feature_points_int, dtype=np.float32)

    affine_mtx = cv2.getAffineTransform(feature_points[:3, :], warped_feature_points_float[:3, :])

    # warped_feature_points_newnew = np.array([[0, 0], [0 + width_int, 0], [0, 0 + height_int], [0 + width_int, 0 + height_int]], dtype=np.float32)
    # affine_mtx = cv2.getAffineTransform(feature_points[:3, :], warped_feature_points_newnew[:3, :])

    warped_image_size = (width_int, height_int)
    output_img = cv2.warpAffine(warped, affine_mtx, dsize=warped_image_size)

    if show_image:
        show("warpedAffine", output_img)

    return output_img


def warpAffine_contour_wrong(original, contour):
    orig_width, orig_height, _ = original.shape
    rectangle = cv2.minAreaRect(contour)
    # The points below are in the configuration of the contour being part of the big "image" (original)
    # When focused on the ROI, all coordinates get translated to the (0, 0) point
    # width and height won't change, angle won't change
    # Although center will change - and be centered around the center of the ROI
    (center, (width, height), angle) = rectangle
    rect_points = np.intp(cv2.boxPoints(rectangle))

    x, y, w, h = cv2.boundingRect(rect_points)
    bounding_rect_contour = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
    roi = original[y : y + h, x : x + w]
    roi_center = (int(w / 2), int(h / 2))

    cv2.circle(roi, center=roi_center, radius=20, color=(0, 0, 0), thickness=cv2.FILLED)
    show("roi", roi)

    rot_matrix = cv2.getRotationMatrix2D(roi_center, -(90 - angle), scale=1)

    # (w, h) : computes the transformation and fits inside a box of dimension (w, h), which is the dimension of the bounding rectangle
    #          hence a rectangle which has greater dimensions that the actual picture
    # (width, height) : computes the transformation to fit in a rectangle that has exactly the size of the picture

    # DOES NOT WORK WELL
    warpaffine = cv2.warpAffine(roi, rot_matrix, (int(height), int(width)))

    # Fits bounding rectangle
    warpaffine = cv2.warpAffine(roi, rot_matrix, (w, h))
    show("Affine Transfo", warpaffine)


def warpPerspective_contour(original, contour):
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

    # show("Output Perspective", output_img)

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


def output(original, picture_name, contours: list, success: bool):
    """
    Takes 3 arguments :
    - the original image
    - the list of contours on that image
    - whether the contours are well suited
    """
    if success == True:

        for idx, contour in enumerate(contours):

            # Option : before wrap
            # output_img = extract_contour(original, contour)

            output_img = warpAffine_contour(original, contour)

            # Option below looses quality
            # output_img = warpPerspective_contour(original, contour)
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
    show("copy contour", copy)
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
