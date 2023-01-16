## Working on mamie0019.jpg

from final import *
from statistics import mean

#####
##
## original, original_with_main_contours, PictureContours, keyboard, message = load_end_to_end(picture_name="mamie0008.jpg")

SMALL_ANGLE_THRESH = 7

original, original_with_main_contours, PictureContours, message = final_steps(picture_name="mamie0008.jpg", THRESH_MIN=245, THESH_MAX=255)
contour = PictureContours[0][0]
test = original.copy()
cv2.drawContours(test, [contour], -1, (0, 255, 0), 40)
show("Contour", test)

# def get_angles(contour):
# funny shape because OpenCV. it's a Nx1 vector of 2-channel elements
# fix that up, remove the silly dimension

num_point = len(contour)
contour.shape = (-1, 2)

# the vectors are differences of coordinates
# a points into the point, b out of the point
a = contour - np.roll(contour, 1, axis=0)
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

point_category = np.transpose([["good"] * num_point])

# this is just for printing/displaying, not useful in code
print("point and angle:")
enriched_contour = np.hstack([contour, angles_degrees.reshape((-1, 1)), alengths.reshape((-1, 1)), blengths.reshape((-1, 1)), point_category])
print(enriched_contour)


canvas_rows = 6000
canvas_columns = 6000

canvas = np.zeros((canvas_rows, canvas_columns, 3))  # floats, range 0..1
canvas_original = canvas.copy()

cv2.polylines(canvas, [contour], isClosed=True, color=(1, 1, 1))

# fontScale=20 // thickness=5 : thickness is good, fontScale is way too big
# fontScale=7 // thickness=5 :

for i, angle in enumerate(angles_degrees):
    cv2.circle(canvas, center=tuple(contour[i]), radius=20, color=(0, 0, 1), thickness=cv2.FILLED)
    cv2.putText(canvas, f"{angle:+.1f}", org=tuple(contour[i]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=7, color=(0, 1, 1), thickness=5)

show("Canvas", canvas)


## Identify bad points vs points for scission between

# Maximum length of the sides of the polygon
MAX_SIDE_LENGTH = alengths.max()
THRESHOLD = 0.25


scission_information = []
scission_dict = dict()

for index, point in enumerate(enriched_contour):
    x_coord = point[0]
    y_coord = point[1]
    angle = abs(float(point[2]))
    a_line = float(point[3])
    b_line = float(point[4])
    # Index -1 is to determine if this is a good point, a scission point, or a bad point
    length_thresh = MAX_SIDE_LENGTH * THRESHOLD
    if angle < SMALL_ANGLE_THRESH:
        if a_line > length_thresh and b_line > length_thresh:
            enriched_contour[index][-1] = "scission"
            scission_dict["scission_point"] = list([x_coord, y_coord])
            scission_dict["before_scission_point"] = list([enriched_contour[index - 1][0], enriched_contour[index - 1][1]])
            scission_dict["after_scission_point"] = list([enriched_contour[index + 1][0], enriched_contour[index + 1][1]])
            scission_information.append(scission_dict)
        else:
            enriched_contour[index][-1] = "bad"


if len(scission_information) >= 1:
    scission = scission_information[0]
    before = np.asarray(scission["before_scission_point"], dtype=int)
    after = np.asarray(scission["after_scission_point"], dtype=int)
    middle = [mean([before[0], after[0]]), mean([before[1], after[1]])]


cv_rows = 6000
cv_columns = 6000

cv = np.zeros((cv_rows, cv_columns, 3))  # floats, range 0..1
cv2.polylines(cv, [contour], isClosed=True, color=(1, 1, 1))

for i, angle in enumerate(angles_degrees):
    if enriched_contour[i][-1] == "bad":
        cv2.circle(cv, center=tuple(contour[i]), radius=20, color=(0, 0, 1), thickness=cv2.FILLED)
    elif enriched_contour[i][-1] == "scission":
        cv2.circle(cv, center=tuple(contour[i]), radius=20, color=(1, 0, 0), thickness=cv2.FILLED)
    else:
        cv2.circle(cv, center=tuple(contour[i]), radius=20, color=(0, 1, 0), thickness=cv2.FILLED)
# This prints the point which is in the middle of the 2 scission points
cv2.circle(cv, center=tuple(middle), radius=20, color=(42, 35, 9), thickness=cv2.FILLED)


show("Canvas", cv)


if __name__ == "__main__":

    original, original_with_main_contours, PictureContours, message = final_steps(picture_name="mamie0008.jpg", THRESH_MIN=245, THESH_MAX=255)
    contour = PictureContours[0][0]
    test = original.copy()
    # cv2.drawContours(test, contour_0, -1, (0, 255, 0), 40) # This only shows points
    cv2.drawContours(test, [contour], -1, (0, 255, 0), 40)
    show("Contour", test)
    enriched_contour = get_angles(contour)

    ### Taken directly from : https://stackoverflow.com/questions/68971669/calculate-inner-angles-of-a-polygon
