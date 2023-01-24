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


def haar_model(img, picture_name, rotation, model=face_default_cascade, draw=None):
    """
    Test with rotations

    #imutils works well
    rot = imutils.rotate(picture.img, angle=90) # imutils crops the image in the rotation

    # https://docs.opencv.org/3.4/d1/de5/classcv_1_1CascadeClassifier.html#aaf8181cb63968136476ec4204ffca498
    """

    # rotated = rotate_np(img, k)
    img = resize(img, 0.5)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # faces = cascade.detectMultiScale(gray, 1.1, minNeighbors=5, minSize=(200, 200))
    faces = model.detectMultiScale(gray, 1.1, minNeighbors=5, minSize=(40, 40))
    x, y, w, h = 0, 0, 0, 0

    img_copy = img.copy()
    # pdb.set_trace()

    summary = []

    for index, (x, y, w, h) in enumerate(faces):
        summary.append([w * h, 0])
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


def dnn_model(img, picture_name, rotation, model=YUNET_PATH, draw=None):
    # https://opencv.org/opencv-face-detection-cascade-classifier-vs-yunet/
    # Code sample : https://gist.github.com/UnaNancyOwen/3f06d4a0d04f3a75cc62563aafbac332 from
    #     https://medium.com/@silkworm/yunet-ultra-high-performance-face-detection-in-opencv-a-good-solution-for-real-time-poc-b01063e251d5
    #     https://docs.opencv.org/4.5.4/d0/dd4/tutorial_dnn_face.html
    detector = cv2.FaceDetectorYN.create(YUNET_PATH, "", (320, 320), SCORE_THRESHOLD, NMS_THRESHOLD, TOP_K)  # this will be changed

    img = resize(img, 0.5)
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

        summary.append([area, float(confidence)])

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
    show(f"{picture_name} - Detected faces", img_copy)
    summary.sort(key=lambda x: x[0], reverse=True)
    return summary


def get_all_faces_areas(img, picture_name, func, **kwargs):
    faces_areas_per_rotation = {"k": [], "rotation": [], "areas": []}
    for k in range(4):
        rotated_img = rotate_np(img, k)
        faces_areas = func(rotated_img, k, picture_name, **kwargs)
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


def get_correct_rotation_new(faces_areas_per_rotation):
    """
    Method can completely be challenged...
    Will pretend 3 faces should be detected. If len < 3, just add 0s to the list
    Average the size of the 3 biggest areas
    """
    face_areas = faces_areas_per_rotation["areas"]
    summary = []
    for faces_rotation in face_areas:
        # print(faces_rotation)
        # print(len(faces_rotation))
        weighted_cum_area = summary.append(sum([face[0] * face[1] for face in faces_rotation]))
        # print(weighted_cum_area)

    index_correct_rotation = summary.index(max(summary))
    correct_k = faces_areas_per_rotation["k"][index_correct_rotation]
    return correct_k


def all_steps(picture_name):
    img = load_original(picture_name, dir="cropped")
    faces_areas_per_rotation = get_all_faces_areas(img, picture_name, dnn_model, draw=True)
    print(faces_areas_per_rotation)
    correct_k = get_correct_rotation_new(faces_areas_per_rotation)
    print(f"{picture_name} - Rotation needed : {correct_k} * 90°")
    rotated_img = rotate_np(img, correct_k)
    show(f"Rotated img - {picture_name} - {correct_k} * 90", rotated_img)


def log_results(picture_name):
    img = load_original(picture_name, dir="cropped")
    picture_rotation = {}
    for k in range(4):
        rotated_img = rotate_np(img, k)
        cv2.imshow(f"{picture_name} - Rotation : {k} degrees", rotated_img)
        r = cv2.waitKey()
        if r == 27 or r == 32:  # Stopping with escape of space bar
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        elif r == ord("o"):  # o is for OK
            print(f"Valid rotation for {picture_name} is {k} * 90 degrees")
            picture_rotation[picture_name] = k
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            break
    rotation_file = PROJECT_DIR + "rotation_metadata.csv"
    headers = ["picture_name", "num_rotations_needed"]
    if path.exists(rotation_file) is False:
        with open(rotation_file, "a") as wr:
            writ = csv.writer(wr)
            print("Printing the headers")
            writ.writerow(headers)
    with open(rotation_file, "a") as w:
        writer = csv.writer(w)
        for key, value in picture_rotation.items():
            writer.writerow([key, value])


if __name__ == "__main__":
    # from test_detection import *
    # img = load_original("mamie0039_03.jpg", dir="cropped")
    # pictures = "mamie0010_02.jpg" # Works
    # pictures = "mamie0039_03.jpg"

    """
    pictures = "mamie0004_02.jpg"
    pictures = [
        "mamie0036_03.jpg",
        "mamie0038_01.jpg",
        "mamie0039_01.jpg",
        "mamie0039_03.jpg",
        "mamie0010_01.jpg",
        "mamie0010_02.jpg",
        "mamie0010_03.jpg",
        "mamie0004_02.jpg",
    ]

    if type(pictures) == list:
        for picture in pictures:
            all_steps(picture)
    else:
        all_steps(pictures)
    """

    pictures_to_process = sorted(os.listdir(CROPPED_DIR))[:50]
    for file in pictures_to_process:
        if file.endswith(".jpg") or file.endswith(".png"):
            log_results(file)
            # all_steps(file)
    pass
