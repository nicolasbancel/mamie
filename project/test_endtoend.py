from transfo import *
from contour import *
from testing import *
from constant import *

PROCESSING_DIR = "/Users/nicolasbancel/git/perso/mamie/project/images/processing/canny_tutorial/"

if __name__ == "__main__":

    white_rectangles = True
    add_margin = True
    thresholding = False
    canny_detection = True

    original = load_original()
    if white_rectangles:
        original = whiten_edges(original)
    if add_margin:
        original = add_borders(original)

    img_grey = grey_original(original)
    img_blur = cv2.blur(img_grey, (3, 3))

    if thresholding:
        img_blur = cv2.adaptiveThreshold(
            img_grey,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            blockSize=3,
            C=2,
        )

    detected_edges = cv2.Canny(img_blur, 12, 12 * 3, 3)
    # cv2.imwrite(f"{PROCESSING_DIR}canny_edges_lowthresh_{val}_{args.input}", detected_edges)

    ### CONTOURS SECTION ###

    retrieval_mode = cv2.RETR_EXTERNAL
    retrieval_method = cv2.CHAIN_APPROX_NONE
    num_contours = 10

    contours, hierarchy = cv2.findContours(detected_edges, retrieval_mode, retrieval_method)
    original_with_contours = original.copy()
    main_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:num_contours]
    cv2.drawContours(original_with_contours, main_contours, -1, (0, 255, 0), 30)
    print("Number of contours identified: ", len(contours))
    show("Original with contours", original_with_contours)

    ### END OF CONTOURS SECTION ###

    contours, _ = find_contours(source=detected_edges)
    original_with_main_contours, PictureContours, key = draw_main_contours(
        original,
        contours,
        num_contours=6,
        contour_size=40,
        contours_color=(0, 255, 0),
        only_rectangles=False,
        show_image=True,
    )
