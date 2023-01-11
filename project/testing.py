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

TITLE_EROSION_WINDOW = "Erosion"
TITLE_DILATATION_WINDOW = "Dilatation"
TITLE_KERNEL_SIZE = "Kernel Size :"
# EROSION_SIZE = 0

MAX_KERNEL_SIZE = 21


def add_margin(picture_name):
    original = load_original(picture_name)
    original_rectangles = whiten_edges(original, 15, 25, color=(255, 255, 255), show_image=False)
    original_with_border = add_borders(original_rectangles, color=(255, 255, 255), show_image=False)
    return original_with_border


def test_image_prep(picture_name):
    img = add_margin(picture_name)
    img_grey = grey_original(img)
    return (img, img_grey)


def test_threshold(picture_name, method="bilateral", show_image=True):

    img = add_margin(picture_name)
    img_grey = grey_original(img)

    if method == "bilateral":
        "Coming from https://www.youtube.com/watch?v=SQ3D1tlCtNg&t=235s&ab_channel=GiovanniCode"
        bilateral = cv2.bilateralFilter(img_grey, 20, 30, 30)
        edged = cv2.Canny(bilateral, 10, 20)
        bilateral_3 = np.stack((bilateral,) * 3, axis=-1)
        edged_3 = np.stack((edged,) * 3, axis=-1)
        img_hor = np.hstack((img, bilateral_3, edged_3))

        if show_image:
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

        output = {"bilateral": bilateral, "edged": edged}

    elif method == "kmeans":
        Z = img.reshape((-1, 3))
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 5
        ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((img.shape))
        if show_image:
            show("Res2", res2)

        output = {"kmeans": res2}

    elif method == "morph_iterations":
        print("Adaptive thresholds - with or without blur")
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

        # BEST MODEL SO FAR. NO BLUR
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

        stacked = np.hstack((th0_blur, th1_blur, th0, th1))

        labels = ["Adaptive - Mean - Blurred", "Adaptive - Gaussian - Blurred", "Adaptive - Mean - Origin", "Adaptive - Gaussian - Origin"]
        stacked = label_stack(stacked, img_grey, labels, blackwhite=True)

        write(f"adaptive_thresholds_{picture_name}", stacked, folder="processing")

        if show_image:
            show("Stacked", stacked)

        output = {"mean_blurred": th0_blur, "gaussian_blurred": th1_blur, "mean": th0, "gaussian": th1}

        # th0_blur_chan = np.stack((th0_blur,) * 3, axis=-1)
        # th1_blur_chan = np.stack((th1_blur,) * 3, axis=-1)
        # th0_chan = np.stack((th0,) * 3, axis=-1)
        # th1_chan = np.stack((th1,) * 3, axis=-1)

        # stacked = np.hstack((th0_blur_chan, th1_blur_chan, th0_chan, th1_chan))
        # show("Stacked", stacked)

        return (img, output)


def erosion(img, morph_shape=cv2.MORPH_RECT):
    erosion_size = cv2.getTrackbarPos(TITLE_KERNEL_SIZE, TITLE_EROSION_WINDOW)
    # erosion_shape = morph_shape(cv2.getTrackbarPos(title_trackbar_element_shape, title_erosion_window))
    element = cv2.getStructuringElement(morph_shape, (2 * erosion_size + 1, 2 * erosion_size + 1), (erosion_size, erosion_size))
    eroded = cv2.erode(img, element)
    show(TITLE_EROSION_WINDOW, eroded)


def dilatation(img, morph_shape=cv2.MORPH_RECT):
    dilatation_size = cv2.getTrackbarPos(TITLE_KERNEL_SIZE, TITLE_DILATATION_WINDOW)
    # erosion_shape = morph_shape(cv2.getTrackbarPos(title_trackbar_element_shape, title_erosion_window))
    element = cv2.getStructuringElement(morph_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1), (dilatation_size, dilatation_size))
    dilated = cv2.dilate(img, element)
    show(TITLE_DILATATION_WINDOW, dilated)


def test_morph_operations(img):
    # Erosion
    cv2.namedWindow(TITLE_EROSION_WINDOW)
    # 4 is for the value at start
    # Max kernel size is for the max value kernel size can have
    cv2.createTrackbar(TITLE_KERNEL_SIZE, TITLE_EROSION_WINDOW, 4, MAX_KERNEL_SIZE, erosion)

    erosion(img)


if __name__ == "__main__":
    original, output = test_threshold("mamie0037.jpg", method="morph_iterations", show_image=False)
    picked_output = output["mean"]
    show("Mean - Not blurred", picked_output)

    # Morph changes
    # test_morph_operations(picked_output)

    # By default : retrieval_mode=cv2.RETR_EXTERNAL
    contours, _ = find_contours(source=picked_output, retrieval_mode=cv2.RETR_LIST)
    original_with_main_contours, PictureContours, key = draw_main_contours(
        original,
        contours,
        num_contours=6,
        contour_size=40,
        contours_color=(0, 255, 0),
        only_rectangles=True,
        show_image=True,
    )
