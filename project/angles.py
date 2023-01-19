## Working on mamie0019.jpg

## Working on mamie0008.jpg - complex one because small angles also


## Working on mamie0030.jpg as well

## mamie0024.jpg : works perfectly

# from final import *
from utils import *
from statistics import mean
from shapely.geometry import Polygon, LineString
from shapely.ops import linemerge, unary_union, polygonize
from shapely import Point, MultiPoint
from sympy import symbols, Eq, solve
import geopandas as gpd  # ONLY NEEDED IN DEV ENVIRONMENT
import pdb

#####
##
## original, original_with_main_contours, PictureContours, keyboard, message = load_end_to_end(picture_name="mamie0008.jpg")


def get_angles(contour):
    contour.shape = (-1, 2)
    a = contour - np.roll(contour, 1, axis=0)
    b = np.roll(a, -1, axis=0)
    alengths = np.linalg.norm(a, axis=1)
    blengths = np.linalg.norm(b, axis=1)
    crossproducts = np.cross(a, b) / alengths / blengths
    angles = np.arcsin(crossproducts)
    angles_degrees = angles / np.pi * 180

    return angles_degrees, alengths, blengths


def plot_angles(contour, angles_degrees):
    canvas_rows = 6000
    canvas_columns = 6000
    canvas = np.zeros((canvas_rows, canvas_columns, 3))  # floats, range 0..1
    # OR OTHER SOLUTION
    # new_canvas = (canvas * 255).astype(np.uint8)
    # OR
    canvas_copy = np.uint8(canvas)
    canvas_original = canvas.copy()
    cv2.polylines(canvas_copy, [contour], isClosed=True, color=(1, 1, 1))
    for i, angle in enumerate(angles_degrees):
        cv2.circle(canvas_copy, center=tuple(contour[i]), radius=20, color=(0, 0, 1), thickness=cv2.FILLED)
        cv2.putText(canvas_copy, f"{angle:+.1f}", org=tuple(contour[i]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=7, color=(0, 1, 1), thickness=5)

    # UNCOMMENT FOR TESTING
    # show("Canvas", canvas_copy)


def enrich_contour_info(contour, angles_degrees, alengths, blengths, exclude_bad_points=True):
    num_point = len(contour)
    max_side_length = alengths.max()
    point_category = np.transpose([["good"] * num_point])
    contour_ = np.hstack([contour, angles_degrees.reshape((-1, 1)), alengths.reshape((-1, 1)), blengths.reshape((-1, 1)), point_category])

    scission_dict = dict()
    scission_information = []

    area = cv2.contourArea(contour)

    # set_trace()

    for index, point in enumerate(contour_):
        # print(index, point)
        angle = abs(float(point[2]))
        a_line = float(point[3])
        b_line = float(point[4])
        # Index -1 is to determine if this is a good point, a scission point, or a bad point
        length_thresh = max_side_length * THRESHOLD
        if angle < SMALL_ANGLE_THRESH:
            if a_line > length_thresh and b_line > length_thresh and area > MAX_AREA_THRESHOLD:
                # Long line with small angles on a small area are now considered bad points
                contour_[index][-1] = "scission"
            else:
                contour_[index][-1] = "bad"

    enriched_contour = []
    idx_to_remove = []

    # Exclusion or not of the bad points of the contour

    if exclude_bad_points == True:
        # remove bad points from enriched_contour
        for index, pt in enumerate(contour_):
            if pt[5] == "bad":
                idx_to_remove.append(index)
            else:
                enriched_contour.append(pt)
        # remove bad points from angle_degrees
        # https://stackoverflow.com/questions/11303225/how-to-remove-multiple-indexes-from-a-list-at-the-same-time

    else:
        enriched_contour = contour_

    ## WATCH OUT !!! angle_degress DOES NOT HAVE THE SAME SHAPE AS enriched_contour SINCE WE'VE DELETED SOME POINTS FROM
    ## ENRICHED CONTOURS, BUT NOT IN angle_degrees

    # Depending on the above : now possible to really identify well the closest points of the scission points
    # If the scission point is surrounded by some bad points, depending on exclude_bad_points, the output will vary

    for idx, element in enumerate(enriched_contour):
        point_type = element[5]
        if point_type == "scission":
            x_coord = element[0]
            y_coord = element[1]
            scission_dict["scission_point"] = list([x_coord, y_coord])
            scission_dict["before_scission_point"] = list([enriched_contour[idx - 1][0], enriched_contour[idx - 1][1]])
            next_idx = (idx + 1) % num_point  # If scission is at index 6 in contour of shape 7 : makes next index 0 instead of 7 (out of range)
            scission_dict["after_scission_point"] = list([enriched_contour[next_idx][0], enriched_contour[next_idx][1]])
            scission_information.append(scission_dict)

    # Capturing the scission point, once data is clean

    if len(scission_information) >= 1:
        scission = scission_information[0]
        scission_point = np.asarray(scission["scission_point"], dtype=int)
        before = np.asarray(scission["before_scission_point"], dtype=int)
        after = np.asarray(scission["after_scission_point"], dtype=int)
        middle_point = [mean([before[0], after[0]]), mean([before[1], after[1]])]
    else:
        # This scenario can happen : when the area is identified as a very big area
        # Although there's no scission point - it's just either a big picture, or 2 pictures, parallel, which have been
        # regrouped into the same big rectangle
        middle_point = None
        scission_point = None

    # pdb.set_trace()

    return enriched_contour, scission_information, middle_point, scission_point, max_side_length


def from_enriched_to_regular(enriched_contour):
    contour = []

    for point in enriched_contour:
        x = int(point[0])
        y = int(point[1])
        contour.append([x, y])
    contour = np.array(contour, dtype=np.int64)

    return contour


def plot_points(enriched_contour, middle_point):
    # Shows bad points anyways since no cleaning has been done yet
    cv_rows = 6000
    cv_columns = 6000
    # pdb.set_trace()
    contour = from_enriched_to_regular(enriched_contour)

    cv = np.zeros((cv_rows, cv_columns, 3))  # floats, range 0..1
    cv2.polylines(cv, [contour], isClosed=True, color=(1, 1, 1))

    for i, angle in enumerate(enriched_contour):
        if enriched_contour[i][-1] == "bad":
            cv2.circle(cv, center=tuple(contour[i]), radius=20, color=(0, 0, 1), thickness=cv2.FILLED)
        elif enriched_contour[i][-1] == "scission":
            cv2.circle(cv, center=tuple(contour[i]), radius=20, color=(1, 0, 0), thickness=cv2.FILLED)
        else:
            cv2.circle(cv, center=tuple(contour[i]), radius=20, color=(0, 1, 0), thickness=cv2.FILLED)
    # This prints the point which is in the middle of the 2 scission points
    if middle_point is not None:
        cv2.circle(cv, center=tuple(middle_point), radius=20, color=(42, 35, 9), thickness=cv2.FILLED)

    # UNCOMMENT FOR TESTING
    # show("Canvas", cv)

    return cv


def find_extrapolation(middle_point, scission_point, max_side_length):
    """
    For a given massive contour (interpreted as a polygon), with a scission point
    The function determines the scission line associated, which would split the polygon into 2 parts
    2 steps :
    - Builds the intersection line - finds its parameters / equation
    - Identifies the intersection point between the scission line and the polygon
    - Splits the polygon
    """
    # pdb.set_trace()
    line = LineString([middle_point, scission_point])

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
    eq = Eq(((k * xc + m) - yb) ** 2 + (xc - xb) ** 2 - max_side_length**2, 0)
    solutions = solve(eq)

    xc1 = solutions[0]
    yc1 = k * xc1 + m

    c1 = [int(xc1), int(yc1)]

    xc2 = solutions[1]
    yc2 = k * xc2 + m

    c2 = [int(xc2), int(yc2)]

    ## There are 2 solutions to the equation (2 end points)
    # One "after" the scission point, in the opposite side of the middle point :
    #   - That's the one we want to keep : it will intersect with the polygon
    # One "before" the scission point, in the same direction as the middle point
    #   - That one will not intersect with the polygon : it will, but just at the scission point, which is redundant)

    # Reference vector is thus the direction of middle -> scission
    # We want the scission -> extrapolated to follow the same direction
    vector_ref = scission_point - middle_point
    vector_c1 = c1 - scission_point
    vector_c2 = c2 - scission_point

    if np.dot(vector_ref, vector_c1) > 0:
        extrapolated_point = c1
    else:
        extrapolated_point = c2

    return extrapolated_point


def split_contour(contour, extrapolated_point, scission_point, middle_point, original, canvas=None):
    """
    - Determine whether or not the line intersects the polygon
        - Documentation : https://stackoverflow.com/questions/6050392/determine-if-a-line-segment-intersects-a-polygon
    - Determine the intersection point + draws the splitting line
    - Splits the polygon in halft
    """
    new_line = LineString([scission_point, extrapolated_point])
    polygon = Polygon(contour)

    # FIND THE INTERSECTION

    intersections = new_line.intersection(polygon)

    # pdb.set_trace()

    # print(f"Type of the intersection : {type(intersections)}")

    if type(intersections) == LineString:
        # if type(intersections) == MultiPoint:
        # if len(intersections) >= 2:

        # There are 2 intersections. 1 is the scission point (since it's the starting point).
        # The other is the interesting point

        first_intersection = np.array([intersections.boundary.geoms[0].x, intersections.boundary.geoms[0].y], dtype=int)
        second_intersection = np.array([intersections.boundary.geoms[1].x, intersections.boundary.geoms[1].y], dtype=int)

        if (first_intersection == scission_point).all():
            intersection_point = second_intersection
        else:
            intersection_point = first_intersection

        if canvas is not None:
            cv2.line(canvas, scission_point, extrapolated_point, (0, 255, 0), thickness=7)
            cv2.circle(canvas, center=tuple(second_intersection), radius=20, color=(42, 35, 9), thickness=cv2.FILLED)
            show("Canvas", canvas)

        splitting_line = LineString([middle_point, intersection_point])

        # SPLIT THE POLYGON

        # SPLITTING LINE IS USELESS, NO ? VS NEW_LINE...
        # merged = linemerge([polygon.boundary, splitting_line])
        merged = linemerge([polygon.boundary, new_line])
        borders = unary_union(merged)
        polygons = list(polygonize(borders))

        # pdb.set_trace()

        # split_contours is a list of contours
        # Polygons in Shapely repeat the first and last coordinate point
        # We should avoid that - otherwise it counts 1 corner twice - hence we would get 5 corners recorded for a rectangle
        split_contours = [np.array(list(pol.exterior.coords)[:-1], dtype=int) for pol in polygons]
    else:
        # This happens in scenario "mamie0047.jpg" where there's a scission point
        # This scission point is on the complete edge of the polygon, has a long length, and
        # does not intersect with the polygon
        intersection_point = None
        split_contours = [contour]

    color_list = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]

    original_copy = original.copy()

    for index, p in enumerate(split_contours):
        # print(f"Number of points in contour : {len(p)}")
        # print(p)
        cv2.drawContours(original_copy, [p], -1, color_list[index], 40)
        for point in p:
            cv2.circle(original_copy, center=tuple(point), radius=20, color=(1, 0, 0), thickness=cv2.FILLED)

    # UNCOMMENT FOR TESTING
    # show("New contours", original_copy)

    return split_contours, intersection_point


def fix_contours(main_contours, original):
    test = original.copy()
    final_image = original.copy()
    final_contours = []
    color_index = 0

    for contour in main_contours:
        contour_area = cv2.contourArea(contour)
        # print(contour_area, f"vs {MAX_AREA_THRESHOLD}")
        cv2.drawContours(test, [contour], -1, (0, 255, 0), 40)
        # UNCOMMENT FOR TESTING
        # show("Contour", test)
        # GETTING ANGLES
        angles_degrees, alengths, blengths = get_angles(contour)
        # print(f"Number of points in contour : {len(contour)}")
        # print(angles_degrees)
        # plot_angles(contour, angles_degrees)
        enriched_contour, scission_information, middle_point, scission_point, max_side_length = enrich_contour_info(contour, angles_degrees, alengths, blengths)
        cv = plot_points(enriched_contour, middle_point)
        if scission_point is not None:
            # print("Contour needs to be splitted")
            extrapolated_point = find_extrapolation(middle_point, scission_point, max_side_length)
            clean_contour = from_enriched_to_regular(enriched_contour)
            new_contours, intersection_point = split_contour(clean_contour, extrapolated_point, scission_point, middle_point, original)
        else:
            # print(f"Contour has good shape - no need for split - color index = {color_index}")
            # Reduce size of contour
            # new_contours has to be a list - in this case, it's a list of 1 single element
            clean_contour = from_enriched_to_regular(enriched_contour)
            # At least, it removes the bad points
            new_contours = [clean_contour]
        for cont in new_contours:
            # print(f"Contour is too big - color index = {color_index}")
            final_contours.append(cont)
            draw(final_image, cont, color_index)
            color_index += 1
        # elif contour_area > MIN_AREA_THRESHOLD and contour_area <= MAX_AREA_THRESHOLD:
        # print(f"Contour has good shape - no need for split - color index = {color_index}")
        # Reduce size of contour
        # contour = contour[:, 0, :]
        # final_contours.append(contour)
        # draw(final_image, contour, color_index)
        # color_index += 1

    # UNCOMMENT FOR TESTING
    # show("Final contours", final_image)

    return final_contours, final_image


if __name__ == "__main__":
    # Circular because final_steps contains fix_contours
    # original, original_with_main_contours, PictureContours, message = final_steps(picture_name="mamie0024.jpg", THRESH_MIN=245, THESH_MAX=255)
    pass
