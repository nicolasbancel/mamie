import cv2


def find_contours(source):
    contours, hierarchy = cv2.findContours(
        source.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    return contours, hierarchy


def draw_contours(original, contours, show_image=False):
    original_with_contours = original.copy()
    cv2.drawContours(original_with_contours, contours, -1, (0, 255, 0), 30)
    # print("Number of contours identified: ", len(contours))
    if show_image:
        cv2.imshow("Output", original_with_contours)
        cv2.waitKey()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    return contours, original_with_contours


def draw_main_contours(
    original,
    contours,
    num_contours,
    contour_size,
    contours_color,
    only_rectangles=True,
    show_image=True,
):
    main_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:num_contours]

    PictureContours = []
    num_rectangles = 0

    for c in main_contours:
        ### Approximating the contour
        # Calculates a contour perimeter or a curve length
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)
        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        screenCnt = approx
        # print(f'Number of corners : {len(approx)}')
        if len(approx) == 4:
            num_rectangles += 1

        if only_rectangles:
            if len(approx) == 4:
                PictureContours.append(screenCnt)
        else:
            PictureContours.append(screenCnt)
        # show the contour (outline)

    original_with_main_contours = original.copy()

    cv2.drawContours(
        original_with_main_contours, PictureContours, -1, contours_color, contour_size
    )

    cv2.imshow("Output - Biggest contours", original_with_main_contours)
    print("Number of contours identified: ", len(contours))
    print(f"Out of {num_contours} biggest contours - {num_rectangles} are rectangles")

    if show_image:
        cv2.waitKey()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    return original_with_main_contours, PictureContours
