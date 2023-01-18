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

SMALL_ANGLE_THRESH = 7
THRESHOLD = 0.25
MAX_AREA_THRESHOLD = 10000000

# MIN_AREA_THRESHOLD = 6000000 WAS CAUSING PROBLEMS FOR POLAROIDs - THEY WERE TOO SMALL
MIN_AREA_THRESHOLD = 5000000


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


def enrich_contour_info(contour, angles_degrees, alengths, blengths):
    num_point = len(contour)
    max_side_length = alengths.max()
    point_category = np.transpose([["good"] * num_point])
    enriched_contour = np.hstack([contour, angles_degrees.reshape((-1, 1)), alengths.reshape((-1, 1)), blengths.reshape((-1, 1)), point_category])

    scission_dict = dict()
    scission_information = []

    # set_trace()

    for index, point in enumerate(enriched_contour):
        # print(index, point)
        x_coord = point[0]
        y_coord = point[1]
        angle = abs(float(point[2]))
        a_line = float(point[3])
        b_line = float(point[4])
        # Index -1 is to determine if this is a good point, a scission point, or a bad point
        length_thresh = max_side_length * THRESHOLD
        if angle < SMALL_ANGLE_THRESH:
            if a_line > length_thresh and b_line > length_thresh:
                if index + 1 >= num_point:
                    # This covers corner case where last point in contour is the scission point
                    # In this case, capturing the info about the next point corresponds to getting 0th point of the contour
                    # mamie0047.jpg : contour has 7 points. Scission is at index = 6. max index is enriched_contour[6]
                    next_index = 0
                else:
                    next_index = index + 1
                enriched_contour[index][-1] = "scission"
                scission_dict["scission_point"] = list([x_coord, y_coord])
                scission_dict["before_scission_point"] = list([enriched_contour[index - 1][0], enriched_contour[index - 1][1]])
                scission_dict["after_scission_point"] = list([enriched_contour[next_index][0], enriched_contour[next_index][1]])
                scission_information.append(scission_dict)
            else:
                enriched_contour[index][-1] = "bad"

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

    return enriched_contour, scission_information, middle_point, scission_point, max_side_length


def plot_points(angle_degrees, enriched_contour, contour, middle_point):
    cv_rows = 6000
    cv_columns = 6000

    cv = np.zeros((cv_rows, cv_columns, 3))  # floats, range 0..1
    cv2.polylines(cv, [contour], isClosed=True, color=(1, 1, 1))

    for i, angle in enumerate(angle_degrees):
        if enriched_contour[i][-1] == "bad":
            cv2.circle(cv, center=tuple(contour[i]), radius=20, color=(0, 0, 1), thickness=cv2.FILLED)
        elif enriched_contour[i][-1] == "scission":
            cv2.circle(cv, center=tuple(contour[i]), radius=20, color=(1, 0, 0), thickness=cv2.FILLED)
        else:
            cv2.circle(cv, center=tuple(contour[i]), radius=20, color=(0, 1, 0), thickness=cv2.FILLED)
    # This prints the point which is in the middle of the 2 scission points
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

        # new_contours is a list of contours
        new_contours = [np.array(list(pol.exterior.coords), dtype=int) for pol in polygons]
    else:
        # This happens in scenario "mamie0047.jpg" where there's a scission point
        # This scission point is on the complete edge of the polygon, has a long length, and
        # does not intersect with the polygon
        intersection_point = None
        new_contours = [contour]

    color_list = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]

    original_copy = original.copy()

    for index, p in enumerate(new_contours):
        # print(f"Number of points in contour : {len(p)}")
        # print(p)
        cv2.drawContours(original_copy, [p], -1, color_list[index], 40)
        for point in p:
            cv2.circle(original_copy, center=tuple(point), radius=20, color=(1, 0, 0), thickness=cv2.FILLED)

    # UNCOMMENT FOR TESTING
    # show("New contours", original_copy)

    return new_contours, intersection_point


def fix_contours(PictureContours, original):
    test = original.copy()
    final_image = original.copy()
    final_contours = []
    color_index = 0

    for contour_info in PictureContours:
        contour = contour_info[0]
        contour_area = contour_info[1]
        if contour_area > MAX_AREA_THRESHOLD:
            cv2.drawContours(test, [contour], -1, (0, 255, 0), 40)
            # UNCOMMENT FOR TESTING
            # show("Contour", test)

            # GETTING ANGLES
            angles_degrees, alengths, blengths = get_angles(contour)
            # plot_angles(contour, angles_degrees)
            enriched_contour, scission_information, middle_point, scission_point, max_side_length = enrich_contour_info(
                contour, angles_degrees, alengths, blengths
            )
            # pdb.set_trace()
            if scission_point is not None:
                cv = plot_points(angles_degrees, enriched_contour, contour, middle_point)
                extrapolated_point = find_extrapolation(middle_point, scission_point, max_side_length)
                # new_contours, intersection_point = split_contour(contour, extrapolated_point, scission_point, middle_point, original, cv)
                new_contours, intersection_point = split_contour(contour, extrapolated_point, scission_point, middle_point, original)
            else:
                # new_contours has to be a list - in this case, it's a list of 1 single element
                new_contours = [contour]
            for cont in new_contours:
                # print(f"Contour is too big - color index = {color_index}")
                final_contours.append(cont)
                draw(final_image, cont, color_index)
                color_index += 1
        elif contour_area > MIN_AREA_THRESHOLD and contour_area <= MAX_AREA_THRESHOLD:
            # print(f"Contour has good shape - no need for split - color index = {color_index}")
            # Reduce size of contour
            contour = contour[:, 0, :]
            final_contours.append(contour)
            draw(final_image, contour, color_index)
            color_index += 1

    # UNCOMMENT FOR TESTING
    # show("Final contours", final_image)

    return final_contours, final_image


if __name__ == "__main__":
    # Circular because final_steps contains fix_contours
    # original, original_with_main_contours, PictureContours, message = final_steps(picture_name="mamie0024.jpg", THRESH_MIN=245, THESH_MAX=255)
    pass
