## Working on mamie0019.jpg
## Working on mamie0030.jpg as well

from final import *
from statistics import mean
from shapely.geometry import Polygon, LineString
from shapely.ops import linemerge, unary_union, polygonize
from sympy import symbols, Eq, solve
import geopandas as gpd  # ONLY NEEDED IN DEV ENVIRONMENT

#####
##
## original, original_with_main_contours, PictureContours, keyboard, message = load_end_to_end(picture_name="mamie0008.jpg")

SMALL_ANGLE_THRESH = 7


def get_angles(contour):
    contour.shape = (-1, 2)
    a = contour - np.roll(contour, 1, axis=0)
    b = np.roll(a, -1, axis=0)
    alengths = np.linalg.norm(a, axis=1)
    blengths = np.linalg.norm(b, axis=1)
    MAX_SIDE_LENGTH = alengths.max()
    crossproducts = np.cross(a, b) / alengths / blengths
    angles = np.arcsin(crossproducts)
    angles_degrees = angles / np.pi * 180

    return angles_degrees, MAX_SIDE_LENGTH


def plot_angles(contour, angles_degrees, num_point):
    canvas_rows = 6000
    canvas_columns = 6000
    canvas = np.zeros((canvas_rows, canvas_columns, 3))  # floats, range 0..1
    # OR OTHER SOLUTION
    # new_canvas = (canvas * 255).astype(np.uint8)
    # OR
    # canvas_copy = np.uint8(canvas)
    canvas_original = canvas.copy()
    cv2.polylines(canvas, [contour], isClosed=True, color=(1, 1, 1))
    for i, angle in enumerate(angles_degrees):
        cv2.circle(canvas, center=tuple(contour[i]), radius=20, color=(0, 0, 1), thickness=cv2.FILLED)
        cv2.putText(canvas, f"{angle:+.1f}", org=tuple(contour[i]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=7, color=(0, 1, 1), thickness=5)

    show("Canvas", canvas)


def enrich_contour_info(contour, angles_degrees):
    num_point = len(contour)
    point_category = np.transpose([["good"] * num_point])
    enriched_contour = np.hstack([contour, angles_degrees.reshape((-1, 1)), alengths.reshape((-1, 1)), blengths.reshape((-1, 1)), point_category])

    scission_dict = dict()
    scission_information = []

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
        scission_point = np.asarray(scission["scission_point"], dtype=int)
        before = np.asarray(scission["before_scission_point"], dtype=int)
        after = np.asarray(scission["after_scission_point"], dtype=int)
        middle_point = [mean([before[0], after[0]]), mean([before[1], after[1]])]

    return enriched_contour, scission_information


def plot_points(angle_degrees, enriched_contour, contour):
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
    cv2.circle(cv, center=tuple(middle_point), radius=20, color=(42, 35, 9), thickness=cv2.FILLED)

    show("Canvas", cv)

if __name__ == '__main__':
    
    original, original_with_main_contours, PictureContours, message = final_steps(picture_name="mamie0008.jpg", THRESH_MIN=245, THESH_MAX=255)
    contour = PictureContours[0][0]
    test = original.copy()
    # cv2.drawContours(test, contour_0, -1, (0, 255, 0), 40) # This only shows points
    cv2.drawContours(test, [contour], -1, (0, 255, 0), 40)
    show("Contour", test)
    enriched_contour = get_angles(contour)


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

    # print("angles in degrees:")
    # print(angles_degrees)

    point_category = np.transpose([["good"] * num_point])

    # this is just for printing/displaying, not useful in code
    # print("point and angle:")
    enriched_contour = np.hstack([contour, angles_degrees.reshape((-1, 1)), alengths.reshape((-1, 1)), blengths.reshape((-1, 1)), point_category])
    # print(enriched_contour)


    canvas_rows = 6000
    canvas_columns = 6000

    canvas = np.zeros((canvas_rows, canvas_columns, 3))  # floats, range 0..1
    canvas_original = canvas.copy()

    # https://stackoverflow.com/questions/19103933/depth-error-in-2d-image-with-opencv-python
    canvas_copy = np.uint8(canvas)
    new_canvas = (canvas * 255).astype(np.uint8)

    cv2.polylines(new_canvas, [contour], isClosed=True, color=(1, 1, 1))

    # fontScale=20 // thickness=5 : thickness is good, fontScale is way too big
    # fontScale=7 // thickness=5 :

    for i, angle in enumerate(angles_degrees):
        cv2.circle(new_canvas, center=tuple(contour[i]), radius=20, color=(0, 0, 1), thickness=cv2.FILLED)
        cv2.putText(new_canvas, f"{angle:+.1f}", org=tuple(contour[i]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=7, color=(0, 1, 1), thickness=5)

    show("Canvas", new_canvas)


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
        scission_point = np.asarray(scission["scission_point"], dtype=int)
        before = np.asarray(scission["before_scission_point"], dtype=int)
        after = np.asarray(scission["after_scission_point"], dtype=int)
        middle_point = [mean([before[0], after[0]]), mean([before[1], after[1]])]


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
    cv2.circle(cv, center=tuple(middle_point), radius=20, color=(42, 35, 9), thickness=cv2.FILLED)


    show("Canvas", cv)

    ###### FINDING INTERSECTION

    # https://stackoverflow.com/questions/47359985/shapely-intersection-point-between-line-and-polygon-in-3d
    # https://stackoverflow.com/questions/62388276/how-to-create-an-infinite-line-from-two-given-points-to-get-intersection-with-ot

    line = LineString([middle_point, scission_point])
    polygon = Polygon(contour)

    # Line has equation
    # Y = k*x + m

    # a : middle
    # b : scission point

    xa = middle_point[0]
    ya = middle_point[1]

    xb = scission_point[0]
    yb = scission_point[1]

    k = (yb - ya) / (xb - xa)
    m = yb - k * xb

    # Finding a coordinate Yc in the opposite direction of middle point. Which, starting from Scission point,
    # has a norm that's equal to the max of the polygon line - which "ensures" there'll be an intersection
    # yc
    # xc

    xc = symbols("xc")
    eq = Eq(((k * xc + m) - yb) ** 2 + (xc - xb) ** 2 - MAX_SIDE_LENGTH**2)
    solutions = solve(eq)

    xc1 = solutions[0]
    yc1 = k * xc1 + m

    c1 = [int(xc1), int(yc1)]

    xc2 = solutions[1]
    yc2 = k * xc2 + m

    c2 = [int(xc2), int(yc2)]


    # cv2.circle(cv, center=tuple(c1), radius=20, color=(42, 35, 9), thickness=cv2.FILLED)
    # cv2.circle(cv, center=tuple(c2), radius=20, color=(42, 35, 9), thickness=cv2.FILLED)

    # Vector : scission - middle
    # Need to keep the vector that goes in the same direction as vector_ref (scission - middle)

    vector_ref = scission_point - middle_point
    vector_c1 = c1 - scission_point
    vector_c2 = c2 - scission_point

    if np.dot(vector_ref, vector_c1) > 0:
        extrapolated_point = c1
    else:
        extrapolated_point = c2

    new_line = LineString([scission_point, extrapolated_point])

    cv2.line(cv, scission_point, extrapolated_point, (0, 255, 0), thickness=7)
    show("Canvas", cv)

    # Determine whether or not the line intersects the polygon
    # https://stackoverflow.com/questions/6050392/determine-if-a-line-segment-intersects-a-polygon
    # new_line.intersects(polygon) : True


    intersections = new_line.intersection(polygon)
    # There are 2 intersections. 1 is the scission point (since it's the starting point).
    # The other is the interesting point

    first_intersection = np.array([intersections.boundary.geoms[0].x, intersections.boundary.geoms[0].y], dtype=int)
    # first_intersection == scission_point
    second_intersection = np.array([intersections.boundary.geoms[1].x, intersections.boundary.geoms[1].y], dtype=int)

    if (first_intersection == scission_point).all():
        intersection_point = second_intersection
    else:
        intersection_point = first_intersection

    # intersection_point = np.asarray([inter.x, inter.y], dtype=int)

    cv2.circle(cv, center=tuple(second_intersection), radius=20, color=(42, 35, 9), thickness=cv2.FILLED)

    # inter = line.intersection(polygon)

    show("Canvas", cv)

    ## SPLITTING THE POLYGON INTO 2 PARTS

    splitting_line = LineString([middle_point, intersection_point])  ## This should split the polygon in 2 (and interesect with it)
    merged = linemerge([polygon.boundary, splitting_line])
    borders = unary_union(merged)
    polygons = list(polygonize(borders))

    polygons_array = [np.array(list(pol.exterior.coords), dtype=int) for pol in polygons]

    # Green, then red, then blue
    color_list = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]

    original_copy = original.copy()

    for index, p in polygons_array:
        cv2.drawContours(original_copy, [p], -1, color_list[index], 40)


    # cv2.drawContours(original_copy, [first_contour], -1, color_list[0], 40)


if __name__ == "__main__":

    

    ### Taken directly from : https://stackoverflow.com/questions/68971669/calculate-inner-angles-of-a-polygon

    ### Test
