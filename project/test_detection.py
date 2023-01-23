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

picture_name = "mamie0010_01.jpg"
img = load_original(picture_name, dir="cropped")

face_default_cascade = cv2.CascadeClassifier(os.path.join(OPENCV_DATA_DIR, "haarcascade_frontalface_default.xml"))

face_alt_tree_cascade = cv2.CascadeClassifier(os.path.join(OPENCV_DATA_DIR, "haarcascade_frontalface_alt_tree.xml"))
face_alt_cascade = cv2.CascadeClassifier(os.path.join(OPENCV_DATA_DIR, "haarcascade_frontalface_alt.xml"))
profileface_cascade = cv2.CascadeClassifier(os.path.join(OPENCV_DATA_DIR, "haarcascade_profileface.xml"))


def rotate_np(img, k):
    rotated = np.rot90(img, k=k)
    return rotated


def get_faces(img, k, cascade=face_default_cascade):
    """
    Test with rotations

    #imutils works well
    rot = imutils.rotate(picture.img, angle=90) # imutils crops the image in the rotation

    # https://docs.opencv.org/3.4/d1/de5/classcv_1_1CascadeClassifier.html#aaf8181cb63968136476ec4204ffca498
    """

    rotated = rotate_np(img, k)
    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    # faces = cascade.detectMultiScale(gray, 1.1, minNeighbors=5, minSize=(200, 200))
    faces = cascade.detectMultiScale(gray, 1.1, minNeighbors=5, minSize=(40, 40))
    x, y, w, h = 0, 0, 0, 0

    rotated_cv_copy = rotated.copy()
    # pdb.set_trace()

    faces_area = []

    for index, (x, y, w, h) in enumerate(faces):
        faces_area.append(w * h)
        cv2.rectangle(rotated_cv_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(rotated_cv_copy, (x + int(w * 0.5), y + int(h * 0.5)), 4, (0, 255, 0), -1)
        cv2.putText(
            rotated_cv_copy,
            f"Face N°{index} - Dims {w}x{h}pix",
            org=(x, y - 20),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 255, 0),
            thickness=2,
        )
    cv2.putText(rotated_cv_copy, f"Log of rotation {k} * 90°", org=(100, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(0, 0, 255), thickness=4)
    show("Img with faces", rotated_cv_copy)

    return sorted(faces_area, reverse=True)


def get_all_faces_areas(img, cascade):
    faces_areas_per_rotation = {"k": [], "rotation": [], "areas": []}
    for k in range(4):
        faces_areas = get_faces(img, k, cascade)
        faces_areas_per_rotation["k"].append(k)
        faces_areas_per_rotation["rotation"].append(int(k * 90))
        faces_areas_per_rotation["areas"].append(faces_areas)
    return faces_areas_per_rotation


def resize(img, scale):
    new_width = int(img.shape[1] * scale)
    new_height = int(img.shape[0] * scale)
    return cv2.resize(img, (new_width, new_height))


SCORE_THRESHOLD = 0.9
NMS_THRESHOLD = 0.3
TOP_K = 20


def dnn_model(model=YUNET_PATH):
    detector = cv2.FaceDetectorYN.create(YUNET_PATH, "", (320, 320), SCORE_THRESHOLD, NMS_THRESHOLD, TOP_K)  # this will be changed


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
    pass
