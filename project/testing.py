from transfo import *
from contour import *

# picture_name = "mamie0001.jpg"


# Does not identify the separation between 2 pictures
# picture_name = "mamie0036.jpg"

# Does not identify the separation between 2 pictures
# picture_name = "mamie0008.jpg"

# Does not identify the separation between 2 pictures
# AND influenced by white color (bad contour)
# picture_name = "mamie0037.jpg"


def add_margin(picture_name):
    original = load_original(picture_name)
    original_rectangles = whiten_edges(
        original, 15, 25, color=(255, 255, 255), show_image=False
    )
    original_with_border = add_borders(
        original_rectangles, color=(255, 255, 255), show_image=False
    )
    return original_with_border


def image_processing(picture_name):
    img = add_margin(picture_name)
    pass


def test_processing(picture_name, method="bilateral"):

    img = add_margin(picture_name)
    img_grey = grey_original(img)

    if method == "bilateral":
        "Coming from https://www.youtube.com/watch?v=SQ3D1tlCtNg&t=235s&ab_channel=GiovanniCode"
        bilateral = cv2.bilateralFilter(img_grey, 20, 30, 30)
        edged = cv2.Canny(bilateral, 10, 20)
        bilateral_3 = np.stack((bilateral,) * 3, axis=-1)
        edged_3 = np.stack((edged,) * 3, axis=-1)
        img_hor = np.hstack((img, bilateral_3, edged_3))
        show("YouTube video", img_hor)
        contours, _ = find_contours(source=edged)
        original_with_main_contours, PictureContours, key = draw_main_contours(
            img,
            contours,
            num_contours=6,
            contour_size=40,
            contours_color=(0, 255, 0),
            only_rectangles=False,
            show_image=True,
        )

    elif method == "kmeans":
        Z = img.reshape((-1, 3))
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 5
        ret, label, center = cv2.kmeans(
            Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((img.shape))
        show("Res2", res2)

    elif method == "morph_iterations":
        # changed blockSize from 11 to 3

        img_blurred = cv2.medianBlur(img_grey, 5)

        th0_blur = cv2.adaptiveThreshold(
            img_blurred,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            blockSize=3,
            C=2,
        )
        th1_blur = cv2.adaptiveThreshold(
            img_blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=3,
            C=2,
        )

        th0 = cv2.adaptiveThreshold(
            img_grey,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            blockSize=3,
            C=2,
        )
        th1 = cv2.adaptiveThreshold(
            img_grey,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=3,
            C=2,
        )

        th0_blur_chan = np.stack((th0_blur,) * 3, axis=-1)
        th1_blur_chan = np.stack((th1_blur,) * 3, axis=-1)
        th0_chan = np.stack((th0,) * 3, axis=-1)
        th1_chan = np.stack((th1,) * 3, axis=-1)

        stacked = np.hstack((th0_blur_chan, th1_blur_chan, th0_chan, th1_chan))
        show("Stacked", stacked)
