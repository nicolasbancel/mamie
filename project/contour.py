import numpy as np
from utils import *
from utils_contour import *
import cv2
from statistics import mean

CANVAS_ROWS = 6000
CANVAS_COLUMNS = 6000


class Contour:
    def __init__(self, np_array=None):
        self.points = np_array
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

    def plot_points(self):
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

        show("Canvas with regular, scission, bad, and middle points", cv)

        return cv
