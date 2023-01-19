from constant import *
from pathlib import Path
import numpy as np


def output(original, picture_name, contours: list, success: bool):
    """
    Takes 3 arguments :
    - the original image
    - the list of contours on that image
    - whether the contours are well suited
    """
    if success == True:

        for idx, contour in enumerate(contours):
            # print(f"Printing contour # {idx + 1}")
            mask = np.zeros_like(original)
            # List of 1 element. Index -1 is for printing "all" elements of that list
            cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
            out = np.zeros_like(original)
            out[mask == 255] = original[mask == 255]
            # show("out", out)
            # np.where(mask == 255) results in a 3 dimensional array
            (y, x, z) = np.where(mask == 255)
            (topy, topx) = (np.min(y), np.min(x))
            (bottomy, bottomx) = (np.max(y), np.max(x))
            output = out[topy : bottomy + 1, topx : bottomx + 1]

            # in case picture_name is provided as a path
            # filename = Path(picture_name).stem
            (filename, extension) = picture_name.split(".")
            if idx + 1 < 10:
                suffix = "_0" + str(idx + 1)
            else:
                suffix = "_" + str(idx + 1)
            new_filename = filename + suffix + "." + extension
            path = os.path.join(CROPPED_DIR, new_filename)

            cv2.imwrite(path, output)


if __name__ == "__main__":
    picture_name = "mamie0001.jpg"
    """
    # The code below cannot be executed since final_steps is already using the "export" function
    # Serpent qui se mord la queue
    original, original_w_main_contours, original_w_final_contours, main_contours, final_contours, message = final_steps(
        picture_name, THRESH_MIN, THESH_MAX, export="all"
    )
    
    success = message["success"] == True
    export(original, picture_name, final_contours, success)
    """
