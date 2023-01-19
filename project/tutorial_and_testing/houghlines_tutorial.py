# https://stackoverflow.com/questions/45322630/how-to-detect-lines-in-opencv
# https://www.geeksforgeeks.org/line-detection-python-opencv-houghline-method/
# OpencCV HoughLines does not find vertical lines : https://stackoverflow.com/questions/62009971/openccv-houghlines-does-not-find-vertical-lines

from transfo import *
from contour import *
from testing import *
from constant import *
from numpy.linalg import norm

CANNY_DIR = "/Users/nicolasbancel/git/perso/mamie/project/images/processing/canny_tutorial/"
LINES_DIR = "/Users/nicolasbancel/git/perso/mamie/project/images/processing/lines_tutorial/"
# num_lines = 50 # First file

num_lines = 200

original = load_original()
original = whiten_edges(original)
original = add_borders(original)

img_grey = grey_original(original)

img_blur = cv2.blur(img_grey, (3, 3))
detected_edges = cv2.Canny(img_grey, 12, 12 * 3, 3)


# threshold = 200 # doesn't find vertical lines
# threshold = 100 # doesn't find vertical lines

lines = cv2.HoughLines(detected_edges, 1, np.pi / 180, threshold=100)


def sort_key(elem):
    # 3rd element is the line length
    return elem[2]


# line_ex = [(1, 2, 17), (2, 4, 200), (1, 0, 2)]
# sorted(list_ex, key=sort_key, reverse=True)

"""
A = (0, 1)
B = (9, 10)

a = np.array((0, 1))
b = np.array((9, 10))
norm(b-a)
"""

all_lines_info = []

# The below for loop runs till r and theta values
# are in the range of the 2d array
for r_theta in lines:
    arr = np.array(r_theta[0], dtype=np.float64)
    r, theta = arr
    # Stores the value of cos(theta) in a
    a = np.cos(theta)
    # Stores the value of sin(theta) in b
    b = np.sin(theta)
    # x0 stores the value rcos(theta)
    x0 = a * r
    # y0 stores the value rsin(theta)
    y0 = b * r
    # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
    x1 = int(x0 + 1000 * (-b))
    # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
    y1 = int(y0 + 1000 * (a))
    # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
    x2 = int(x0 - 1000 * (-b))
    # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
    y2 = int(y0 - 1000 * (a))
    # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
    # (0,0,255) denotes the colour of the line to be
    # drawn. In this case, it is red.

    # list does not work : [x1, y1]
    # nor tuple : (x1, y1)
    # nor weird [[x1, x2]]
    # print(f"x1 : {x1} {type(x1)} / y1 : {y1} {type(y1)} / x2 : {x2} {type(x2)}/ y2 : {y2} {type(y2)}")
    A = np.array((x1, y1))
    B = np.array((x2, y2))
    line_length = norm(B - A)
    all_lines_info.append((A, B, line_length))

main_lines = sorted(all_lines_info, key=sort_key, reverse=True)[:num_lines]

original_lines = original.copy()

for line in main_lines:
    cv2.line(original_lines, line[0], line[1], (0, 0, 255), 4)

show("Original with lines", original_lines)
# write from transfo file is deprecated
# write("lines_thresh100_mamie0037.jpg", original_lines, folder="processing/lines_tutorial")
