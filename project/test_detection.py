from constant import *
from final import *
from utils import *
from transfo import *

# import imutils
from PIL import Image
import pdb

from PIL import Image, ExifTags

#################################################
## DOCUMENTATION
# https://medium.com/analytics-vidhya/how-to-auto-rotate-the-image-using-deep-learning-c34b2e0e157d
#

# Giving up on cascades : too many false positives


# filepath = '/Users/nicolasbancel/git/perso/mamie/data/mosaic/cropped/mamie0010_01.jpg'

# filepath = '/Users/nicolasbancel/git/perso/mamie/data/mosaic/cropped/mamie0005_03.jpg'
#################################################


face_default_cascade = cv2.CascadeClassifier(os.path.join(OPENCV_DATA_DIR, "haarcascade_frontalface_default.xml"))

face_alt_tree_cascade = cv2.CascadeClassifier(os.path.join(OPENCV_DATA_DIR, "haarcascade_frontalface_alt_tree.xml"))
face_alt_cascade = cv2.CascadeClassifier(os.path.join(OPENCV_DATA_DIR, "haarcascade_frontalface_alt.xml"))
profileface_cascade = cv2.CascadeClassifier(os.path.join(OPENCV_DATA_DIR, "haarcascade_profileface.xml"))

COLOR = (0, 0, 255)  # Red
RECT_THICKNESS = 2
TIP_RADIUS = 3
TIP_THICKNESS = -1  # (filled)
TEXT_XPOS = 0
TEXT_YPOS = -10
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.5
TEXT_THICKNESS = 2

TEXT_TITLE_POS = (100, 100)


def rotate_np(img, k):
    rotated = np.rot90(img, k=k)
    return rotated


def haar_model(img, rotation, model=face_default_cascade, draw=None):
    """
    Test with rotations

    #imutils works well
    rot = imutils.rotate(picture.img, angle=90) # imutils crops the image in the rotation

    # https://docs.opencv.org/3.4/d1/de5/classcv_1_1CascadeClassifier.html#aaf8181cb63968136476ec4204ffca498
    """

    # rotated = rotate_np(img, k)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # faces = cascade.detectMultiScale(gray, 1.1, minNeighbors=5, minSize=(200, 200))
    faces = model.detectMultiScale(gray, 1.1, minNeighbors=5, minSize=(40, 40))
    x, y, w, h = 0, 0, 0, 0

    img_copy = img.copy()
    # pdb.set_trace()

    summary = []

    for index, (x, y, w, h) in enumerate(faces):
        summary.append(
            [
                w * h,
            ]
        )
        if draw is not None:
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
    if draw is not None:
        cv2.putText(img_copy, f"Log of rotation {rotation} * 90°", org=TEXT_TITLE_POS, fontFace=TEXT_FONT, fontScale=3, color=(0, 0, 255), thickness=4)

    show("Img with faces", img_copy)

    # True by biggest picture
    summary.sort(key=lambda x: x[0], reverse=True)
    return summary


SCORE_THRESHOLD = 0.9
NMS_THRESHOLD = 0.3
TOP_K = 20


def dnn_model(img, rotation, model=YUNET_PATH, draw=None):
    # https://opencv.org/opencv-face-detection-cascade-classifier-vs-yunet/
    # Code sample : https://gist.github.com/UnaNancyOwen/3f06d4a0d04f3a75cc62563aafbac332 from
    #     https://medium.com/@silkworm/yunet-ultra-high-performance-face-detection-in-opencv-a-good-solution-for-real-time-poc-b01063e251d5
    #     https://docs.opencv.org/4.5.4/d0/dd4/tutorial_dnn_face.html
    detector = cv2.FaceDetectorYN.create(YUNET_PATH, "", (320, 320), SCORE_THRESHOLD, NMS_THRESHOLD, TOP_K)  # this will be changed
    img = resize(img, 0.5)
    height, width, _ = img.shape
    img_copy = img.copy()
    detector.setInputSize((width, height))
    _, faces = detector.detect(img)

    faces = faces if faces is not None else []

    summary = []

    for face in faces:

        area = face[2] * face[3]
        confidence = "{:.2f}".format(face[-1])

        summary.append([area, confidence])

        if draw is not None:
            box = face[:4].astype(int)
            cv2.rectangle(img_copy, box, COLOR, RECT_THICKNESS, cv2.LINE_AA)

            tips = face[4 : len(face) - 1].astype(int)
            tips = np.array_split(tips, len(tips) / 2)

            for tip in tips:
                cv2.circle(img_copy, tip, TIP_RADIUS, COLOR, TIP_THICKNESS, cv2.LINE_AA)

            position = (box[0] + TEXT_XPOS, box[1] + TEXT_YPOS)
            cv2.putText(img_copy, f"Confidence level: {confidence}", position, TEXT_FONT, TEXT_SCALE, TEXT_THICKNESS)
    if draw is not None:
        cv2.putText(img_copy, f"Log of rotation {rotation} * 90°", org=TEXT_TITLE_POS, fontFace=TEXT_FONT, fontScale=3, color=(0, 0, 255), thickness=4)
    show("Detected faces", img_copy)
    summary.sort(key=lambda x: x[0], reverse=True)
    return summary


def get_all_faces_areas(img, func):
    faces_areas_per_rotation = {"k": [], "rotation": [], "areas": []}
    for k in range(4):
        rotated_img = rotate_np(img, k)
        faces_areas = func(rotated_img, k)
        faces_areas_per_rotation["k"].append(k)
        faces_areas_per_rotation["rotation"].append(int(k * 90))
        faces_areas_per_rotation["areas"].append(faces_areas)
    return faces_areas_per_rotation


def resize(img, scale):
    new_width = int(img.shape[1] * scale)
    new_height = int(img.shape[0] * scale)
    return cv2.resize(img, (new_width, new_height))


def rotate_exif(filepath):
    # https://stackoverflow.com/questions/13872331/rotating-an-image-with-orientation-specified-in-exif-using-python-without-pil-in
    try:
        image = Image.open(filepath)
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == "Orientation":
                break
        exif = dict(image._getexif().items())

        if exif[orientation] == 3:
            image = image.transpose(Image.ROTATE_180)
        elif exif[orientation] == 6:
            image = image.transpose(Image.ROTATE_270)
        elif exif[orientation] == 8:
            image = image.transpose(Image.ROTATE_90)
        image.save(filepath)
        image.close()

    except (AttributeError, KeyError, IndexError) as e:
        print(f"Error : {e}")  # cases: image don't have getexif
        pass


if __name__ == "__main__":
    picture_name = "mamie0010_01.jpg"
    img = load_original(picture_name, dir="cropped")
    get_all_faces_areas(img, dnn_model)
    pass
