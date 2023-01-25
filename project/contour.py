import numpy as np
from utils import *
from utils_contour import *
from Mosaic import *
import cv2
from statistics import mean
from shapely.geometry import Polygon, LineString
from shapely.ops import linemerge, unary_union, polygonize
from shapely import Point, MultiPoint
from sympy import symbols, Eq, solve

CANVAS_ROWS = 6000
CANVAS_COLUMNS = 6000


class Contour:
    def __init__(self, np_array=None):
        self.points = np_array
        self.num_points = len(np_array)
        self.area = cv2.contourArea(np_array)
        self.get_angles()
        self.enrich_contour()

    pass

    def get_angles(self):
        self.points.shape = (-1, 2)
        a = self.points - np.roll(self.points, 1, axis=0)
        b = np.roll(a, -1, axis=0)
        alengths = np.linalg.norm(a, axis=1)
        blengths = np.linalg.norm(b, axis=1)
        crossproducts = np.cross(a, b) / alengths / blengths
        angles_radians = np.arcsin(crossproducts)
        angles = angles_radians / np.pi * 180
        self.angles = angles  # in degrees
        self.alengths = alengths  # Length of the segment on 1 side of the angle
        self.blengths = blengths  # Lenght of the other segment on the other side of the angle

        return angles, alengths, blengths

    def plot_angles(self):
        # Canvas is a black background
        canvas = np.zeros((CANVAS_ROWS, CANVAS_COLUMNS, 3))  # floats, range 0..1
        canvas_copy = np.uint8(canvas)
        # Polygon drawn in white
        cv2.polylines(canvas_copy, [self.points], isClosed=True, color=(255, 255, 255), thickness=2)
        for i, angle in enumerate(self.angles):
            # Point and angle font in red
            cv2.circle(canvas_copy, center=tuple(self.points[i]), radius=20, color=(0, 0, 255), thickness=cv2.FILLED)
            cv2.putText(canvas_copy, f"{angle:+.1f}", org=tuple(self.points[i]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=7, color=(0, 0, 255), thickness=5)

        show("Canvas with angles", canvas_copy)

    def enrich_contour(self, exclude_bad_points=None):
        num_points_contour = len(self.points)
        max_side_length = self.alengths.max()
        point_category = np.transpose([["good"] * num_points_contour])
        contour_ = np.hstack([self.points, self.angles.reshape((-1, 1)), self.alengths.reshape((-1, 1)), self.blengths.reshape((-1, 1)), point_category])

        area = cv2.contourArea(self.points)

        for index, point in enumerate(contour_):
            angle = abs(float(point[2]))
            a_line = float(point[3])
            b_line = float(point[4])
            # Index -1 is to determine if this is a good point, a scission point, or a bad point
            length_thresh = max_side_length * THRESHOLD
            if angle < SMALL_ANGLE_THRESH:
                if a_line > length_thresh and b_line > length_thresh and area > MAX_AREA_THRESHOLD and index not in (0, num_points_contour - 1):
                    # Long line with small angles on a small area are now considered bad points
                    contour_[index][-1] = "scission"
                else:
                    # First and last point of the contour cannot be scission points
                    # Since those are the extreme points of the contour (corners)
                    contour_[index][-1] = "bad"

        enriched_contour = []
        idx_to_remove = []

        # Exclusion or not of the bad points of the contour

        if exclude_bad_points == True:
            # remove bad points from enriched_contour
            for index, pt in enumerate(contour_):
                if pt[5] == "bad":
                    idx_to_remove.append(index)
                    # And point is not being added to enriched_contour (hence, it is removed)
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
        # num_points is the number of points in the new contour (where the bad points may have been removed)
        num_point = len(enriched_contour)
        scission_information = []

        # declaring scission_dict outside the for loop just appends the latest scission information
        # https://stackoverflow.com/questions/35906411/list-on-python-appending-always-the-same-value

        for idx, element in enumerate(enriched_contour):
            scission_dict = dict()
            point_type = element[5]
            if point_type == "scission":
                x_coord = element[0]
                y_coord = element[1]
                # If scission is at index 6 in contour of shape 7 : makes next index 0 instead of 7 (out of range)
                next_idx = (idx + 1) % num_point
                scission_dict = {
                    "scission_point": list([int(x_coord), int(y_coord)]),
                    "before_scission_point": list([int(enriched_contour[idx - 1][0]), int(enriched_contour[idx - 1][1])]),
                    "after_scission_point": list([int(enriched_contour[next_idx][0]), int(enriched_contour[next_idx][1])]),
                }
                scission_information.append(scission_dict)
                # print(scission_information)

        self.enriched = enriched_contour
        self.max_side_length = max_side_length

        # Capturing the scission point, once data is clean

        if len(scission_information) >= 1:
            for i in range(len(scission_information)):
                scission = scission_information[i]
                scission_point = np.asarray(scission["scission_point"], dtype=int)
                before = np.asarray(scission["before_scission_point"], dtype=int)
                after = np.asarray(scission["after_scission_point"], dtype=int)
                middle_point = [mean([before[0], after[0]]), mean([before[1], after[1]])]
                scission["middle_point"] = middle_point
        else:
            # This scenario can happen : when the area is identified as a very big area
            # Although there's no scission point - it's just either a big picture, or 2 pictures, parallel, which have been
            # regrouped into the same big rectangle
            self.middle_point = None
            self.scission_point = None

        self.scission_information = scission_information

        ########################################################
        # IF TREATING ONLY 1 POINT
        # THIS SHOULD BE REMOVED
        # CODE IS NOT CONFIGURED TO BE ABLE TO DEAL WITH MULTIPLE SCISSION POINTS ON A MOSAIC
        ########################################################

        if len(scission_information) >= 1:
            self.middle_point = np.asarray(scission_information[0]["middle_point"], dtype=int)
            self.scission_point = np.asarray(scission_information[0]["scission_point"], dtype=int)

        return self.enriched, self.scission_information, self.middle_point, self.scission_point, self.max_side_length

    def plot_points(self, show=None):
        # Shows bad points anyways since no cleaning has been done yet

        contour = from_enriched_to_regular(self.enriched)

        cv = np.zeros((CANVAS_ROWS, CANVAS_COLUMNS, 3))  # floats, range 0..1
        cv2.polylines(cv, [contour], isClosed=True, color=(255, 255, 255), thickness=2)

        for i, angle in enumerate(self.enriched):
            if self.enriched[i][-1] == "bad":
                cv2.circle(cv, center=tuple(contour[i]), radius=20, color=(0, 0, 255), thickness=cv2.FILLED)
            elif self.enriched[i][-1] == "scission":
                cv2.circle(cv, center=tuple(contour[i]), radius=20, color=(255, 0, 0), thickness=cv2.FILLED)
            else:
                cv2.circle(cv, center=tuple(contour[i]), radius=20, color=(0, 255, 0), thickness=cv2.FILLED)
        # This prints the point which is in the middle of the 2 scission points

        if self.scission_information is not None:
            for dict in self.scission_information:
                cv2.circle(cv, center=tuple(dict["middle_point"]), radius=20, color=(42, 35, 9), thickness=cv2.FILLED)

        if show == True:
            show("Canvas with regular, scission, bad, and middle points", cv)

        return cv

    def find_extrapolation(self):
        """
        ONLY WORKS IF self HAS A SCISSION POINT, HENCE self.scission_point is not None
        For a given massive contour (interpreted as a polygon), with a scission point
        The function determines the scission line associated, which would split the polygon into 2 parts
        2 steps :
        - Builds the intersection line - finds its parameters / equation
        - Identifies the intersection point between the scission line and the polygon
        - Splits the polygon
        """
        line = LineString([self.middle_point, self.scission_point])

        # Line has equation
        # Y = k*x + m

        xa = self.middle_point[0]
        ya = self.middle_point[1]

        xb = self.scission_point[0]
        yb = self.scission_point[1]

        if xb != xa:
            k = (yb - ya) / (xb - xa)
            m = yb - k * xb

            # Finding a coordinate Yc in the opposite direction of middle point. Which, starting from Scission point,
            # has a norm that's equal to the max of the polygon line - which "ensures" there'll be an intersection
            # yc
            # xc

            xc = symbols("xc")
            eq = Eq(((k * xc + m) - yb) ** 2 + (xc - xb) ** 2 - self.max_side_length**2, 0)
            solutions = solve(eq)

            xc1 = solutions[0]
            yc1 = k * xc1 + m

            # pdb.set_trace()

            c1 = [int(xc1), int(yc1)]

            xc2 = solutions[1]
            yc2 = k * xc2 + m

            c2 = [int(xc2), int(yc2)]

        else:
            # The scission point is exactly vertical to the middle point
            # equation of the line is x = xb

            yc = symbols("yc")
            eq = Eq((yc - yb) ** 2 - self.max_side_length**2, 0)

            solutions = solve(eq)

            c1 = [int(xb), int(solutions[0])]
            c2 = [int(xb), int(solutions[1])]

        ## There are 2 solutions to the equation (2 end points)
        # One "after" the scission point, in the opposite side of the middle point :
        #   - That's the one we want to keep : it will intersect with the polygon
        # One "before" the scission point, in the same direction as the middle point
        #   - That one will not intersect with the polygon : it will, but just at the scission point, which is redundant)

        # Reference vector is thus the direction of middle -> scission
        # We want the scission -> extrapolated to follow the same direction
        vector_ref = self.scission_point - self.middle_point
        vector_c1 = c1 - self.scission_point
        vector_c2 = c2 - self.scission_point

        if np.dot(vector_ref, vector_c1) > 0:
            extrapolated_point = c1
        else:
            extrapolated_point = c2

        self.extrapolated_point = extrapolated_point

        return self.extrapolated_point

    def split_contour(self, mosaic: Mosaic, canvas=None):
        """
        - Determine whether or not the line intersects the polygon
            - Documentation : https://stackoverflow.com/questions/6050392/determine-if-a-line-segment-intersects-a-polygon
        - Determine the intersection point + draws the splitting line
        - Splits the polygon in halft
        """
        new_line = LineString([self.scission_point, self.extrapolated_point])
        polygon = Polygon(self.points)

        # FIND THE INTERSECTION

        pdb.set_trace()

        intersections = new_line.intersection(polygon)

        # print(f"Type of the intersection : {type(intersections)}")

        if type(intersections) == LineString:
            # if type(intersections) == MultiPoint:
            # if len(intersections) >= 2:

            # There are 2 intersections. 1 is the scission point (since it's the starting point).
            # The other is the interesting point

            first_intersection = np.array([intersections.boundary.geoms[0].x, intersections.boundary.geoms[0].y], dtype=int)
            second_intersection = np.array([intersections.boundary.geoms[1].x, intersections.boundary.geoms[1].y], dtype=int)

            if (first_intersection == self.scission_point).all():
                intersection_point = second_intersection
            else:
                intersection_point = first_intersection

            if canvas is not None:
                cv2.line(canvas, self.scission_point, self.extrapolated_point, (0, 255, 0), thickness=7)
                cv2.circle(canvas, center=tuple(second_intersection), radius=20, color=(42, 35, 9), thickness=cv2.FILLED)
                # show("Canvas", canvas)

            splitting_line = LineString([self.middle_point, intersection_point])

            # SPLIT THE POLYGON

            merged = linemerge([polygon.boundary, new_line])
            borders = unary_union(merged)
            polygons = list(polygonize(borders))

            # split_contours is a list of contours
            # Polygons in Shapely repeat the first and last coordinate point
            # We should avoid that - otherwise it counts 1 corner twice - hence we would get 5 corners recorded for a rectangle
            split_contours = [np.array(list(pol.exterior.coords)[:-1], dtype=int) for pol in polygons]
        else:
            # This happens in scenario "mamie0047.jpg" where there's a scission point
            # This scission point is on the complete edge of the polygon, has a long length, and
            # does not intersect with the polygon
            intersection_point = None
            split_contours = [self.points]

        self.intersection_point = intersection_point
        self.split_contours = split_contours

        original_copy = mosaic.img.copy()

        for index, p in enumerate(split_contours):
            cv2.drawContours(original_copy, [p], -1, COLOR_LIST[index], 40)
            for point in p:
                cv2.circle(original_copy, center=tuple(point), radius=20, color=(1, 0, 0), thickness=cv2.FILLED)

        return split_contours, intersection_point


if __name__ == "__main__":
    # from Contour import *
    # from Mosaic import *
    mosaic_name = "mamie0009.jpg"
    mosaic = Mosaic(mosaic_name)
    find_contours(mosaic, retrieval_mode=cv2.RETR_EXTERNAL)
    draw_main_contours(mosaic)
    first_contour = mosaic.contours_main[0]
    contour = Contour(first_contour)
    contour.plot_angles()
    contour.plot_points()
    contour.find_extrapolation()
    contour.split_contour(mosaic)
