from utils import *


MOSAIC_METADATA = load_metadata(filename="pictures_per_mosaic.csv")


class Mosaic:
    def __init__(self, mosaic_name=None, cv2_array=None):
        self.mosaic_name = mosaic_name
        self.true_num_pictures = int(MOSAIC_METADATA[mosaic_name])
        self.img = load_original(mosaic_name, dir="source")
        self.img_white_edges = self.whiten_edges()
        self.img_white_borders = self.add_borders()
        self.img_grey = self.grey_original()
        self.img_blur = cv2.GaussianBlur(self.img_grey, (3, 3), 0)
        self.img_thresh = self.thresh()
        self.img_w_main_contours = None
        self.img_w_final_contours = None
        self.contours_all = None
        self.contours_main = None
        self.contours_final = None

        self.num_contours_total = None
        self.num_contours_main = None

    def whiten_edges(
        self,
        thickness_vertical=15,
        thickness_horizontal=25,
        color=(255, 255, 255),  # color=(0, 255, 0) for GREEN
        show_image=False,
    ):
        img_copy = self.img.copy()
        num_row, num_col = img_copy.shape[:2]
        top_left = (0, 0)
        bottom_right_vertical = (thickness_vertical, num_row)
        bottom_right_horizontal = (num_col, thickness_horizontal)

        # Adding horizontal rectangle on top
        cv2.rectangle(img_copy, top_left, bottom_right_vertical, color, -1)

        # Adding vertical rectangle on the left
        cv2.rectangle(img_copy, top_left, bottom_right_horizontal, color, -1)

        if show_image:
            show("With white edges", img_copy)

        return img_copy

    def add_borders(self, color=(255, 255, 255), show_image=False):
        # This script assumes we're adding borders to an image where the edges have already been blanked
        # It applies to self.img_white_edges, not to self.img
        source = self.img_white_edges
        border_size = min(int(0.05 * source.shape[0]), int(0.05 * source.shape[1]))
        top = bottom = left = right = border_size
        borderType = cv2.BORDER_CONSTANT
        img_copy = cv2.copyMakeBorder(source, top, bottom, left, right, borderType, None, color)

        if show_image:
            show("With white border", img_copy)

        return img_copy

    def grey_original(self, show_image=False):
        # This script assumes that greying happens after whiten edges + add borders
        img_grey = self.img_white_borders.copy()
        img_grey = cv2.cvtColor(img_grey, cv2.COLOR_BGR2GRAY)
        if show_image:
            show("Grey image", img_grey)
        return img_grey

    def thresh(self, show_image=False):
        """
        This does 2 things :
        - It both blurs the image
        - And it finds the threshold
        """
        img_thresh = cv2.threshold(self.img_blur, THRESH_MIN, THESH_MAX, cv2.THRESH_BINARY_INV)[1]
        if show_image:
            show("Image Threshold", img_thresh)
        return img_thresh


if __name__ == "__main__":
    # from Mosaic import *
    mosaic_name = "mamie0009.jpg"
    mosaic = Mosaic(mosaic_name)
    show("Regular", mosaic.img)
    show("White edges", mosaic.img_white_edges)
    show("White border", mosaic.img_white_borders)
    show("Grey", mosaic.img_grey)
    show("Blur", mosaic.img_blur)
    show("Thresh", mosaic.img_thresh)
