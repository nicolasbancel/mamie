from constant import *
from final import *
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

    def get_correct_k(self):
        self.correct_k = ROTATION_METADATA.get(self.picture_name)

    def haar_model(self, k, model=FACE_DEFAULT_CASCADE, show_steps=None):
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
        img = self.resize(self.img, 0.5)
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

    def dnn_model(self, k, model=YUNET_PATH, show_steps=None):
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

        img = self.resize(self.img, 0.5)
        # resizing loses a bit of precision - for example with :  "mamie0039_03.jpg"
        height, width, _ = img.shape
        img_copy = img.copy()
        detector.setInputSize((width, height))
        _, faces = detector.detect(img)

        faces = faces if faces is not None else []

        summary = []

        for face in faces:

            area = face[2] * face[3]
            confidence = "{:.2f}".format(face[-1])

            box = face[:4].astype(int)
            tips = face[4 : len(face) - 1].astype(int)
            tips = np.array_split(tips, len(tips) / 2)

            landmarks_density = get_point_density(tips)
            summary.append([area, float(confidence), landmarks_density])
            cv2.rectangle(img_copy, box, COLOR, RECT_THICKNESS, cv2.LINE_AA)
            for tip in tips:
                cv2.circle(img_copy, tip, TIP_RADIUS, COLOR, TIP_THICKNESS, cv2.LINE_AA)

            position = (box[0] + TEXT_XPOS, box[1] + TEXT_YPOS)
            cv2.putText(img_copy, f"Confidence level: {confidence}", position, TEXT_FONT, TEXT_SCALE, color=COLOR, thickness=TEXT_THICKNESS)

        cv2.putText(img_copy, f"Log of rotation {k} * 90°", org=TEXT_TITLE_POS, fontFace=TEXT_FONT, fontScale=2, color=(0, 0, 255), thickness=4)
        if show_steps == True:
            show(f"{self.picture_name} - Detected faces", img_copy)
        return summary

    def get_faces_per_rotation(self, func, **kwargs):
        faces_per_rotation = {"k": [], "rotation": [], "summary": []}
        for k in range(4):
            rotated_img = self.rotate_np(self.img, k)
            summary = func(self, k, **kwargs)
            faces_per_rotation["k"].append(k)
            faces_per_rotation["rotation"].append(int(k * 90))
            faces_per_rotation["summary"].append(summary)
        self.faces_per_rotation
        return faces_per_rotation

    def get_faces(self, k):
        """
        Test with rotations

        #imutils works well
        rot = imutils.rotate(picture.img, angle=90) # imutils crops the image in the rotation

        # https://docs.opencv.org/3.4/d1/de5/classcv_1_1CascadeClassifier.html#aaf8181cb63968136476ec4204ffca498
        """

        rotated = self.rotate_np(k)
        gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        # faces = self.face_cascade.detectMultiScale(gray, 1.1, minNeighbors=5, minSize=(60, 60))
        # Example with mamie0010_01.jpg
        faces = self.face_cascade.detectMultiScale(gray, 1.1, minNeighbors=5, minSize=(200, 200))
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
        cv2.putText(
            rotated_cv_copy, f"Log of rotation {k} * 90°", org=(100, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(0, 0, 255), thickness=4
        )
        show("Img with faces", rotated_cv_copy)

        return sorted(faces_area, reverse=True)

    def get_all_faces_areas(self):
        faces_areas_per_rotation = {"k": [], "rotation": [], "areas": []}
        for k in range(4):
            faces_areas = self.get_faces(k)
            faces_areas_per_rotation["k"].append(k)
            faces_areas_per_rotation["rotation"].append(int(k * 90))
            faces_areas_per_rotation["areas"].append(faces_areas)
        return faces_areas_per_rotation

    def get_correct_rotation(self, faces_areas_per_rotation):
        """
        Method can completely be challenged...
        Will pretend 3 faces should be detected. If len < 3, just add 0s to the list
        Average the size of the 3 biggest areas
        """
        face_areas = faces_areas_per_rotation["areas"]

        average = []
        final_face_areas = []
        # print(face_areas)
        for fa in face_areas:
            if len(fa) >= 3:
                final_fa = fa[:3]
                final_face_areas.append(final_fa)
                average.append(sum(final_fa) / len(final_fa))
            elif len(fa) >= 1 and len(fa) < 3:
                # pdb.set_trace()
                fa.extend([0] * (3 - len(fa)))
                final_fa = fa.copy()
                final_face_areas.append(final_fa)
                average.append(sum(final_fa) / len(final_fa))
            else:
                final_fa = [0] * 3
                final_face_areas.append(final_fa)
                average.append(0)
            # if fa = [24] then completes with 3-1=2 zeros : [24, 0, 0]

        index_correct_rotation = average.index(max(average))
        correct_k = faces_areas_per_rotation["k"][index_correct_rotation]
        self.num_needed_rot90 = correct_k
        return correct_k

    def rotate_image(self):
        faces_areas_per_rotation = self.get_all_faces_areas()
        self.get_correct_rotation(faces_areas_per_rotation)
        self.img_rotated = self.rotate_np(self.num_needed_rot90)
        return self.img_rotated


if __name__ == "__main__":

    # mamie0010.jpg is all wrong

    # in ipython :
    from Picture import *

    picture = Picture(picture_name="mamie0010_01.jpg")
    picture.get_all_faces_areas()

    # NEED TO DEBUG THE FACE DETECTION ALGORITHM WHICH DOES NOT WORK PROPERLY

    # Or loading from cv2 array
    # image_mamie = load_original("mamie0009.jpg", dir='source')
    # picture = Picture(cv2_array = image_mamie)

    # picture = Picture("mamie0008_02.jpg") # No rotation needed
    picture = Picture(picture_name="mamie0009_03.jpg")  # 90d needed
    picture.rotate_image()
    show("Rotated", picture.img_rotated)

    # picture.num_needed_rot90

    # faces_areas_per_rotation = picture.get_all_faces_areas()
    # correct_k = picture.get_correct_rotation(faces_areas_per_rotation)
    # picture.rotated = picture.rotate_np(picture.num_needed_rot90)
