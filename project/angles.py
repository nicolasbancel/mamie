## Working on mamie0019.jpg

from transfo import *
from contour import *
from testing import *
from constant import *
from utils import *
from typing import Literal


picture_name = "mamie0019.jpg"
original = load_original(picture_name)
original = whiten_edges(original)
original = add_borders(original)
img_grey = grey_original(original)
img_blur = cv2.bilateralFilter(img_grey, 9, 75, 75)
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
    show_image=True,
)


if __name__ == "__main__":
    picture_name = " mamie0019.jpg"
    original = load_original(picture_name)
    original = whiten_edges(original)
    original = add_borders(original)
    img_grey = grey_original(original)
    img_blur = cv2.bilateralFilter(img_grey, 9, 75, 75)
    thresh = cv2.threshold(img_blur, THRESH_MIN, THESH_MAX, cv2.THRESH_BINARY_INV)[1]
    contours, _ = find_contours(source=thresh)
    original_with_main_contours, PictureContours, keyboard, message = draw_main_contours(
        original,
        contours,
        num_biggest_contours=6,
        contour_size=40,
        contours_color=(0, 255, 0),
        precision_param=0.1,
        only_rectangles=False,
        show_image=True,
    )

    contour_0 = PictureContours[0][0]
    cv2.contourArea(contour_0)
    test = original.copy()
    # cv2.drawContours(test, contour_0, -1, (0, 255, 0), 40) # This only shows points
    cv2.drawContours(test, [contour_0], -1, (0, 255, 0), 40)
    show("Contour", test)

    points = contour_0

    ### Taken directly from : https://stackoverflow.com/questions/68971669/calculate-inner-angles-of-a-polygon

    # funny shape because OpenCV. it's a Nx1 vector of 2-channel elements
    # fix that up, remove the silly dimension
    points.shape = (-1, 2)

    # the vectors are differences of coordinates
    # a points into the point, b out of the point
    a = points - np.roll(points, 1, axis=0)
    b = np.roll(a, -1, axis=0)  # same but shifted

    # we'll need to know the length of those vectors
    alengths = np.linalg.norm(a, axis=1)
    blengths = np.linalg.norm(b, axis=1)

    # we need only the length of the cross product,
    # and we work in 2D space anyway (not 3D),
    # so the cross product can't result in a vector, just its z-component
    crossproducts = np.cross(a, b) / alengths / blengths

    angles = np.arcsin(crossproducts)
    angles_degrees = angles / np.pi * 180

    print("angles in degrees:")
    print(angles_degrees)

    # this is just for printing/displaying, not useful in code
    print("point and angle:")
    print(np.hstack([points, angles_degrees.reshape((-1, 1))]))

    canvas_rows = 6000
    canvas_columns = 6000

    canvas = np.zeros((canvas_rows, canvas_columns, 3))  # floats, range 0..1
    original = canvas.copy()

    cv2.polylines(canvas, [points], isClosed=True, color=(1, 1, 1))

    for i, angle in enumerate(angles_degrees):
        cv2.circle(canvas, center=tuple(points[i]), radius=20, color=(0, 0, 1), thickness=cv2.FILLED)
        cv2.putText(canvas, f"{angle:+.1f}", org=tuple(points[i]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=20, color=(0, 1, 1), thickness=15)

    show("Canvas", canvas)
