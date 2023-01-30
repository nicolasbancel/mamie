from Picture import *
from utils import *
import cv2
from datetime import datetime
from typing import Literal

FACE_DEFAULT_CASCADE = cv2.CascadeClassifier(os.path.join(OPENCV_DATA_DIR, "haarcascade_frontalface_default.xml"))
FACE_ALT_TREE_CASCADE = cv2.CascadeClassifier(os.path.join(OPENCV_DATA_DIR, "haarcascade_frontalface_alt_tree.xml"))
FACE_ALT_CASCADE = cv2.CascadeClassifier(os.path.join(OPENCV_DATA_DIR, "haarcascade_frontalface_alt.xml"))
PROFILEFACE_CASCADE = cv2.CascadeClassifier(os.path.join(OPENCV_DATA_DIR, "haarcascade_profileface.xml"))

COLOR = (0, 0, 255)  # Red
RECT_THICKNESS = 2
TIP_RADIUS = 3
TIP_THICKNESS = -1  # (filled)
TEXT_XPOS = 0
TEXT_YPOS = -10
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 1
TEXT_THICKNESS = 2

TEXT_TITLE_POS = (100, 100)

########################
## DNN MODEL SECTION
########################

SCORE_THRESHOLD = 0.9
NMS_THRESHOLD = 0.3
TOP_K = 20


def haar_model(picture, k, model=FACE_DEFAULT_CASCADE, show_steps=None):
    """
    Args:
        k :         rotation coefficient. If k = 1, img is rotated by 90°. k = 2, img is rotated by 180°
        model:      used for face detection
        show_steps: show the images / steps or not

    Returns:
        summary: list of n list (n = # of faces detected). Each list (face) has 3 elements :
            - area of the detected face
            - confidence level for this face detection
            - density / closeness of the landmark points
    Source:
        https://docs.opencv.org/3.4/d1/de5/classcv_1_1CascadeClassifier.html#aaf8181cb63968136476ec4204ffca498
    """
    img = picture.resize(0.5)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = model.detectMultiScale(gray, 1.1, minNeighbors=5, minSize=(40, 40))
    x, y, w, h = 0, 0, 0, 0

    img_copy = img.copy()

    summary = []

    for index, (x, y, w, h) in enumerate(faces):
        summary.append([w * h, 0, 0])
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(img_copy, (x + int(w * 0.5), y + int(h * 0.5)), 4, (0, 255, 0), -1)
        cv2.putText(
            img_copy,
            f"Face N°{index} - Dims {w}x{h}pix",
            org=(x + TEXT_XPOS, y - TEXT_YPOS),
            fontFace=TEXT_FONT,
            fontScale=TEXT_SCALE,
            color=COLOR,
            thickness=TEXT_THICKNESS,
        )

    cv2.putText(img_copy, f"Log of rotation {k} * 90°", org=TEXT_TITLE_POS, fontFace=TEXT_FONT, fontScale=3, color=(0, 0, 255), thickness=4)
    if show_steps == True:
        show("Img with faces", img_copy)

    # True by biggest picture
    # summary.sort(key=lambda x: x[0], reverse=True)
    return summary


def dnn_model(picture, k, model=YUNET_PATH, show_steps=None):
    """
    Args:
        k :         rotation coefficient. If k = 1, img is rotated by 90°. k = 2, img is rotated by 180°
        model:      used for face detection
        show_steps: show the images / steps or not

    Returns:
        summary: list of n list (n = # of faces detected). Each list (face) has 3 elements :
            - area of the detected face
            - confidence level for this face detection
            - density / closeness of the landmark points
    Sources:
        - https://opencv.org/opencv-face-detection-cascade-classifier-vs-yunet/
        - Code sample : https://gist.github.com/UnaNancyOwen/3f06d4a0d04f3a75cc62563aafbac332 from
        - https://medium.com/@silkworm/yunet-ultra-high-performance-face-detection-in-opencv-a-good-solution-for-real-time-poc-b01063e251d5
        - https://docs.opencv.org/4.5.4/d0/dd4/tutorial_dnn_face.html
    """

    detector = cv2.FaceDetectorYN.create(YUNET_PATH, "", (320, 320), SCORE_THRESHOLD, NMS_THRESHOLD, TOP_K)  # this will be changed
    img = picture.resize(0.5)
    # resizing loses a bit of precision - for example with :  "mamie0039_03.jpg"
    height, width, _ = img.shape
    mid_height = int(height / 2)
    img_copy = img.copy()
    detector.setInputSize((width, height))
    _, faces = detector.detect(img)

    # x1, y1, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm
    # x1, y1, w, h are the top-left coordinates, width and height of the face bounding box
    # {x, y}_{re, le, nt, rcm, lcm} stands for the coordinates of right eye, left eye, nose tip, the right corner and left corner of the mouth respectively.

    def area_above_middle_line(face, middle_y):
        # top_y is the highest point of the bounding rectangle of the face
        top_x = face[0]
        top_y = face[1]
        #
        face_width = face[2]
        face_height = face[3]
        if top_y > middle_y:
            area_above_middle = 0
            area_below_middle = face_width * face_height
            height_above_middle = 0
        else:
            height_above_middle = min(top_y + face_height, middle_y) - top_y
            area_above_middle = height_above_middle * face_width
            area_below_middle = face_width * (face_height - height_above_middle)
        return area_above_middle, (int(top_x), int(top_y), int(face_width), int(height_above_middle))

    faces = faces if faces is not None else []

    summary = []

    # Draw middle line, to determine the area above the middle line
    cv2.line(img_copy, (0, mid_height), (width, mid_height), color=(0, 255, 0), thickness=3)

    for face in faces:

        area = face[2] * face[3]
        confidence = "{:.2f}".format(face[-1])
        box = face[:4].astype(int)

        # Calculating density of landmarks
        tips = face[4 : len(face) - 1].astype(int)
        tips = np.array_split(tips, len(tips) / 2)
        landmarks_density = get_point_density(tips)

        # Area above middle
        area_above_middle, rectangle_above_middle = area_above_middle_line(face, mid_height)
        cv2.rectangle(img_copy, rectangle_above_middle, color=(255, 255, 255), thickness=-1)

        summary.append([area, float(confidence), landmarks_density, area_above_middle])
        cv2.rectangle(img_copy, box, COLOR, RECT_THICKNESS, cv2.LINE_AA)
        for tip in tips:
            cv2.circle(img_copy, tip, TIP_RADIUS, COLOR, TIP_THICKNESS, cv2.LINE_AA)

        position = (box[0] + TEXT_XPOS, box[1] + TEXT_YPOS)
        cv2.putText(img_copy, f"Confidence level: {confidence}", position, TEXT_FONT, TEXT_SCALE, color=COLOR, thickness=TEXT_THICKNESS)

    cv2.putText(img_copy, f"Log of rotation {k} * 90°", org=TEXT_TITLE_POS, fontFace=TEXT_FONT, fontScale=2, color=(0, 0, 255), thickness=4)
    if show_steps == True:
        show(f"{picture.picture_name} - Detected faces", img_copy)
    return summary


def get_faces_per_rotation(picture, func, **kwargs):
    """
    Args:
        picture :   instance of the Picture class
        func:       either dnn_model or haar_model
        show_steps: show the images / steps or not

    Returns:
        dictionnary: for each rotation, provides a list which includes the 4 elements, for each face identified:
        - area
        - confidence
        - density of each face identified
        - area above middle
    """
    faces_per_rotation = {"k": [], "rotation": [], "summary": []}
    for k in range(4):
        rotated_img = picture.rotate_np(k)
        rot_picture = Picture(picture_name=picture.picture_name, cv2_array=rotated_img)
        summary = func(rot_picture, k, **kwargs)
        faces_per_rotation["k"].append(k)
        faces_per_rotation["rotation"].append(int(k * 90))
        faces_per_rotation["summary"].append(summary)
        picture.faces_per_rotation = faces_per_rotation
    return faces_per_rotation


def weight_cum_area(li):
    """
    Args:
        summary : list of summaries of faces on a picture

    Returns :
      A global cumulated area, which is a SUM(area*confidence)
      Gives an idea of much face surface we were able to capture, weigthed by the confidence
    """
    return sum([face[0] * face[1] for face in li])


def get_rotation_model(picture):
    """
    Args:
        picture: picture.faces_per_rotation especially
                 List of faces detected for each rotation.
                 For each list (face), it has the area, the confidence, the density, the area above middle


    Returns :
      For each rotation, returns a summary list :
      - num_ones : # of faces detected on the rotation with 1.00 confidence level (should be MAX)
      - highest_score : 2nd highest score identified on the picture, after 1.00 (should be MAX)
      - avg_density : avg density of landmarks observed on the pictures (density should be low on faces well identified) (SHOULD BE MIN)
      - weighted cumulated area (should be MAX)

      Ranks the rotations by order of the keys listed above
    """
    summaries = picture.faces_per_rotation["summary"]
    result = []

    def num_faces(li):
        """
        li is a list of lists. Each sublist corresponds to 1 rotation. On that rotation, we capture faces, made of
        - an area
        - an accuracy
        The function returns the max number of faces identified across the different rotations
        Typical input : face_areas = faces_areas_per_rotation["areas"]
        """
        max_num_faces = max([len(faces) for faces in li])
        return max_num_faces

    def rotation_summary(l):
        """
        l corresponds to a list of faces identified on 1 picture rotation
        Each element of l is a 2 element list which corresponds to 1 face
        l[0] : area of the rectanle
        l[1] : accuracy of the prediction
        l[2] : density of points
        l[3] : area above middle
        Typical input : face_area[0]

        Output:
        num_ones: number of perfectly identified faces on the rotation
        highest_score: except accuracy = 1, what's the 2nd highest accuracy score. If none (because no face was found, or because only accuracy = 1), defaults to 0
        avg_density : avg density of points observed on faces
        """
        num_ones = 0
        highest_score = 0
        weigthed_densities = []
        weigthed_areas_above_middle = []
        for face in l:
            accuracy = face[1]
            point_density = face[2]
            weighted_density = point_density * accuracy
            weigthed_densities.append(weighted_density)

            # weigthed_area_above_middle
            area_above_middle = face[3]
            weighted_area_above_middle = area_above_middle * accuracy
            weigthed_areas_above_middle.append(weighted_area_above_middle)

            if accuracy >= 0.99:
                # >= 0.99 instead of == 1 fixes mamie0007_03.jpg
                # where Vladimir and Monique are identified with 1.00 x 0.99, and 1.00 and 1.00
                # but 1.00 x 0.99 is cleaner
                num_ones += 1
            else:
                highest_score = max(highest_score, accuracy)

        if len(weigthed_densities) > 0:
            avg_density = sum(weigthed_densities) / len(weigthed_densities)
            avg_area_above_middle = sum(weigthed_areas_above_middle) / len(weigthed_areas_above_middle)
        else:
            # Can't be None
            avg_density = 99  # giving it a bad ranking
            avg_area_above_middle = 0  # giving it a bad ranking

        return num_ones, highest_score, avg_density, avg_area_above_middle

    # We actually don't use the num_faces parameter
    num_faces_identified = num_faces(summaries)

    for index, rotation in enumerate(summaries):
        num_ones, highest_score, avg_density, avg_area_above_middle = rotation_summary(rotation)
        result.append([num_ones, highest_score, avg_area_above_middle, avg_density, weight_cum_area(rotation), index])
    # Sort by :
    # - rotation that has the highest number of faces perfectly identified
    # - if there's tie in the # of perfectly identified faces, what's the next highest accuracy ?
    # - if there's a tie : which rotation has the biggest area above the middle line (assuming a good picture puts the faces on top of the picture)
    # - if there's a tie : which rotation has the lowest density of landmarks on faces ? (usually, there's a good spread of landmarks)
    #   when a face is well identified. Particularly useful for 1 single picture type of photos
    # - if there's a tie there as well : which rotation has the highest area covered ?
    result.sort(key=lambda x: (-1 * x[0], -1 * x[1], -1 * x[2], x[3], -1 * x[4]))
    # The 1st element of summary is now the "best" rotation config. Its index is stored in the last one element
    correct_index_rotation = result[0][-1]
    rot90_predicted_num = picture.faces_per_rotation["k"][correct_index_rotation]
    picture.rot90_predicted_num = rot90_predicted_num
    picture.rot90_summary = result
    return rot90_predicted_num, result


def fill_log(picture, config_num, log_dict: dict):
    """
    This gets translated into a big dictionnary of lists - which will get printed in a log file
    It incrementally adds info to a previous dictionnary

    Args:
        picture : to get the information from
        log : dictionnary of info related to the rotation of 1 picture

    Returns :
        log : with the information of the input picture added
    """

    log_dict["config_num"].append(config_num)
    log_dict["picture_name"].append(picture.picture_name)
    log_dict["rot90_true_num"].append(picture.rot90_true_num)
    log_dict["rot90_predicted_num"].append(picture.rot90_predicted_num)
    log_dict["success"].append(picture.rot90_true_num == picture.rot90_predicted_num)
    log_dict["rot90_summary"].append(picture.rot90_summary)

    return log_dict


def rotate_one(picture, export_rotated=None, show_steps=None):
    """
    All steps to rotate a picture (predict its rotation needed, and apply it)

    Args:
        picture : to get the information from
        show_steps : set to True to see the steps of the detection (results of each rotation)

    Returns :
        rotated pictured
    """
    get_faces_per_rotation(picture, dnn_model, show_steps=show_steps)
    get_rotation_model(picture)
    # print(picture.rot90_summary)
    # print(f"{picture.picture_name} - Rotation needed : {picture.rot90_predicted_num} * 90 deg")
    picture.img_rotated = picture.rotate_np(picture.rot90_predicted_num)
    if show_steps == True:
        show(f"Rotated img - {picture.picture_name} - {picture.rot90_predicted_num} * 90 deg", picture.img_rotated)

    if export_rotated == True:
        path = os.path.join(ROTATED_AUTO_DIR, picture.picture_name)
        cv2.imwrite(path, picture.img_rotated)

    print(f"Rotation done for image: {picture.picture_name} - by {picture.rot90_predicted_num} * 90 deg")


def rotate_all(picture_list=None, num_pic=None, log=None, show_steps=False, export_rotated=None):
    """
    This does the rotation for 1 picture, a list of hardcoded picture, or n pictures in the CROPPED folder

    Args:
        picture_list : if not None, will execute the rotation only for the pictures in that list
        num_pic : if not None, will execute the rotation for the [0:num_pictures] in the cropped folder
        log : if True, fills and pushes the logs to the results_rotation.csv file
        show_steps : if True, shows the steps of face detection for each rotation

    Returns :
        log : with the information of the input picture added
    """
    if picture_list is not None:
        pictures_to_process = picture_list
    elif num_pic is not None:
        pictures_to_process = sorted(os.listdir(CROPPED_DIR))[:num_pic]
    else:
        pictures_to_process = sorted(os.listdir(CROPPED_DIR))
    print(pictures_to_process)
    config_num = str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log_dict = {
        "config_num": [],
        "picture_name": [],
        "rot90_true_num": [],
        "rot90_predicted_num": [],
        "success": [],
        "rot90_summary": [],
    }
    for picture_name in pictures_to_process:
        if picture_name.endswith(".jpg") or picture_name.endswith(".png"):
            picture = Picture(picture_name)
            rotate_one(picture, export_rotated=export_rotated, show_steps=show_steps)
            log_rot = fill_log(picture, config_num, log_dict)
    if log == True:
        log_results(log_rot, "results_rotations.csv")


if __name__ == "__main__":
    # rotate_all(num_pic=10, log=None, show_steps=True)

    rotations_to_debug = [
        "mamie0014_01.jpg",  # Fixed
        "mamie0021_02.jpg",
        "mamie0029_01.jpg",
        "mamie0029_02.jpg",
        "mamie0030_02.jpg",
        "mamie0031_02.jpg",
        "mamie0034_01.jpg",
        "mamie0035_02.jpg",
        "mamie0036_03.jpg",
        "mamie0038_02.jpg",
        "mamie0038_03.jpg",
        "mamie0039_02.jpg",
        "mamie0039_03.jpg",
        "mamie0041_03.jpg",
        "mamie0045_02.jpg",
        "mamie0049_02.jpg",
        "mamie0055_01.jpg",
        "mamie0063_01.jpg",
        "mamie0065_01.jpg",
        "mamie0065_02.jpg",
        "mamie0067_02.jpg",
        "mamie0070_01.jpg",
        "mamie0083_02.jpg",
        "mamie0084_03.jpg",
        "mamie0107_01.jpg",
        "mamie0131_01.jpg",
        "mamie0164_01.jpg",
        "mamie0165_01.jpg",
        "mamie0182_01.jpg",
        "mamie0204_01.jpg",
        "mamie0209_02.jpg",
        "mamie0252_02.jpg",
    ]

    test = ["mamie0070_01.jpg", "mamie0055_01.jpg", "mamie0063_01.jpg", "mamie0049_02.jpg"]  # Aurait pas corrigé

    # If want to rotate, and log the results of the rotations of the list

    rotate_all(picture_list=rotations_to_debug, log=True, show_steps=True)
    # rotate_all(picture_list=test, log=False, show_steps=True)

    # If want to see the steps, execute what's below :

    """
    for pic in rotations_to_debug:
        print(f"Treating rotation of picture : {pic}")
        picture = Picture(picture_name=pic)
        faces_per_rotation = get_faces_per_rotation(picture, dnn_model, show_steps=True)
        print(picture.faces_per_rotation)
        correct_k, result = get_rotation_model(picture)
        print(f"Rot90 summary : {picture.rot90_summary}")
        print(f"Rot90 predicted : {picture.rot90_predicted_num} // Rot90 actual : {picture.rot90_true_num}")
    """
