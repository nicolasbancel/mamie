from constant import *
from final import *
from utils import *
from transfo import *

# import imutils
from PIL import Image
import pdb


class Picture:
    def __init__(self, picture_name):
        self.picture_name = picture_name
        self.img = load_original(picture_name, dir="cropped")
        self.face_cascade = cv2.CascadeClassifier(os.path.join(OPENCV_DATA_DIR, "haarcascade_frontalface_default.xml"))
        # self.eye_cascade = cv2.CascadeClassifier(os.path.join(OPENCV_DATA_DIR, "haarcascade_eye.xml"))

    def rotate_pil(self, rotation_angle=-90):
        """
        This method crops the picture
        """
        new_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(new_img)
        rotated_cv = np.array(im_pil.rotate(rotation_angle).convert("RGB"))
        # converting from RGB to BGR
        # Source : https://stackoverflow.com/questions/14134892/convert-image-from-pil-to-opencv-format
        rotated_cv = rotated_cv[:, :, ::-1]
        # show("Rotated", rotated_cv)
        return rotated_cv

    def rotate_np(self, k):
        """
        https://stackoverflow.com/questions/43892506/opencv-python-rotate-image-without-cropping-sides
        k is a multiplier of # of times to rotate by 90 degrees
        """
        rotated = np.rot90(self.img, k=k)
        # show("Rotated", rotated)
        return rotated

    def get_faces(self, k):
        """
        Test with rotations

        #imutils works well
        rot = imutils.rotate(picture.img, angle=90) # imutils crops the image in the rotation

        # https://docs.opencv.org/3.4/d1/de5/classcv_1_1CascadeClassifier.html#aaf8181cb63968136476ec4204ffca498
        """

        rotated = self.rotate_np(k)
        # rotated = self.rotate_pil(rotation_angle)
        gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, minNeighbors=5, minSize=(60, 60))
        x, y, w, h = 0, 0, 0, 0

        rotated_cv_copy = rotated.copy()
        # pdb.set_trace()

        faces_area = []

        for (x, y, w, h) in faces:
            faces_area.append(w * h)
            cv2.rectangle(rotated_cv_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(rotated_cv_copy, (x + int(w * 0.5), y + int(h * 0.5)), 4, (0, 255, 0), -1)
        cv2.putText(rotated_cv_copy, f"Log of rotation {k} * 90°", org=(50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=4, color=(0, 0, 255), thickness=5)
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
        print(face_areas)
        for fa in face_areas:
            if len(fa) >= 3:
                final_fa = fa[:3]
                final_face_areas.append(final_fa)
                average.append(sum(final_fa) / len(final_fa))
            elif len(fa) >= 1 and len(fa) < 3:
                final_fa = fa.extend([0] * (3 - len(fa)))
                final_face_areas.append(final_fa)
                average.append(sum(final_fa) / len(final_fa))
            else:
                final_fa = [0] * 3
                final_face_areas.append(final_fa)
                average.append(0)
            # if fa = [24] then completes with 3-1=2 zeros : [24, 0, 0]
        pdb.set_trace()


if __name__ == "__main__":
    # in ipython :
    from Picture import *

    picture = Picture("mamie0008_02.jpg")
    faces_areas_per_rotation = picture.get_all_faces_areas()
    picture.get_correct_rotation(faces_areas_per_rotation)
