from transfo import *

img = load_original("mamie0039_03.jpg", dir="cropped")
cv2.imshow("Hello", img)
k = cv2.waitKey()
if k == 27 or k == 32:  # Stopping with escape of space bar
    cv2.destroyAllWindows()
    cv2.waitKey(1)
elif k == ord("o"):  # o is for OK
    print("Image was valid")
    cv2.destroyAllWindows()
    cv2.waitKey(1)
