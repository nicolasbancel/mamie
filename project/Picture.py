from constant import *
from main import *
from utils import *
from transfo import *
from rotate import *

# import imutils
from PIL import Image
import pdb


ROTATION_METADATA = load_metadata(filename="rotation_metadata.csv")

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


class Picture:
    def __init__(self, picture_name=None, cv2_array=None):
        self.picture_name = picture_name
        """
        if picture_name is not None:
            self.img = load_original(picture_name, dir="cropped")
        else:
            self.img = cv2_array
        """
        self.img = cv2_array if cv2_array is not None else load_original(picture_name, dir="cropped")
        self.face_cascade = cv2.CascadeClassifier(os.path.join(OPENCV_DATA_DIR, "haarcascade_frontalface_default.xml"))
        self.faces_per_rotation = None  # Dict : for each rotation, logs all the info needed of all faces identified
        self.rot90_predicted_num = None
        self.rot90_true_num = ROTATION_METADATA.get(self.picture_name, "Unknown")
        self.rot90_summary = None  # List : summary of 3 key metrics per rotation, used to determine the optimal one
        self.img_rotated = None
        # self.eye_cascade = cv2.C
        # ascadeClassifier(os.path.join(OPENCV_DATA_DIR, "haarcascade_eye.xml"))

    def rotate_np(self, k):
        """
        https://stackoverflow.com/questions/43892506/opencv-python-rotate-image-without-cropping-sides
        k is a multiplier of # of times to rotate by 90 degrees
        """
        rotated = np.rot90(self.img, k=k)
        return rotated

    def resize(self, scale):
        new_width = int(self.img.shape[1] * scale)
        new_height = int(self.img.shape[0] * scale)
        return cv2.resize(self.img, (new_width, new_height))


if __name__ == "__main__":

    # from Picture import *

    # Loading from file
    picture1 = Picture(picture_name="mamie0010_01.jpg")
    attrs1 = vars(picture1)
    print(", \n".join("%s: %s" % item for item in attrs1.items()))

    # Loading from array
    img2 = load_original("mamie0009.jpg", dir="source")
    picture2 = Picture(cv2_array=img2)
    attrs2 = vars(picture2)
    print(", \n".join("%s: %s" % item for item in attrs2.items()))
