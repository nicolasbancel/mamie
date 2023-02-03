from Picture import *
from Mosaic import *
from crop import *
from rotate import *
import cv2


def log_coordinates(event, x, y, flags, params):
    global COORDS
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"x : {x} - y : {y}")
        print(f"({x},{y})")
        print(f"Flags : {flags}")
        print(f"Params : {params}")
        COORDS.append([x, y])
        print(f"coords : {COORDS}")
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(new_img, str(x) + "," + str(y), (x, y), font, 5, (255, 0, 0), 2)


def contour_manual(mosaic):
    global COORDS
    img = mosaic.img_source
    new_img = img.copy()
    # Initializing the COORDS variable once the mosaic is loaded
    COORDS = []
    cv2.imshow("Picture to investigate", new_img)
    cv2.setMouseCallback("Picture to investigate", log_coordinates)
    cv2.imshow("Picture to investigate", new_img)
    r = cv2.waitKey()
    if r == ord("o"):  # o for OK - means we're done with the contouring
        all_points = np.array(COORDS, dtype=int)
        chunk_size = 4
        if len(all_points) % chunk_size == 0:
            contours = [all_points[i : i + chunk_size] for i in range(0, len(all_points), chunk_size)]
            cv2.drawContours(new_img, contours, -1, (0, 255, 0), CONTOUR_SIZE)
        else:
            print("Manual contouring not done correctly : 4 corners per pictures are needed")
            contours = []
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    elif r == 27 or r == 32:  # Escape or space - stopping program
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        print(f"Contouring not completed correctly for {mosaic_name}")
        contours = []
    show(f"Mosaic {mosaic.mosaic_name} with all contours", new_img)
    mosaic.img_w_main_contours = new_img
    mosaic.img_w_final_contours = new_img
    mosaic.contours_main = contours
    mosaic.contours_final = contours
    mosaic.num_contours_final = len(contours)
    mosaic.success = True
    return contours, img


def rotate_manual(picture, save_pic=None):
    """
      Logs the true rotation needed for a picture
      It displays the picture, can go through the 4 90° rotations
      - If the rotation is wrong, user presses the space or ESC key
      - If the rotation is correct, user presses the "o" key (like OK)


    Args:
        picture : picture evaluated
        save_pic : if True : overwrites the picture with its rotated version in the target folder
    """
    for k in range(4):
        rotated_img = picture.rotate_np(k)
        picture_rotation = "Not provided"
        cv2.imshow(f"{picture.picture_name} - Rotation : {k} degrees", rotated_img)
        r = cv2.waitKey()
        if r == 27 or r == 32:  # Stopping with escape of space bar
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        elif r == ord("o"):  # o is for OK
            print(f"Valid rotation for {picture.picture_name} is {k} * 90 degrees")
            picture_rotation = k
            if save_pic == True:
                success_path = os.path.join(ROTATED_MANUAL_DIR, picture.picture_name)
                cv2.imwrite(success_path, rotated_img)
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            break
    rotation_file = PROJECT_DIR + "rotation_metadata.csv"
    headers = ["picture_name", "num_rotations_needed"]
    # Create file + headers if file does not exist
    if path.exists(rotation_file) is False:
        with open(rotation_file, "a") as wr:
            writ = csv.writer(wr)
            print("Printing the headers")
            writ.writerow(headers)
    # We should write to the rotation_metadata.csv file in 2 scenarios:
    # 1. the correct rotation of a picture_name has not been entered
    # 2. the rotation of a picture_name was entered, but we're saying it's actually different :
    #   PROBLEM OF SCENARIO #2 : it duplicates the key for that picture_name in the csv
    if picture.picture_name not in list(ROTATION_METADATA.keys()) or (
        picture.picture_name in list(ROTATION_METADATA.keys()) and ROTATION_METADATA[picture.picture_name] != picture_rotation
    ):
        # Limitation of this : if picture has different rotation, it appends it (hence it is duplicated in the file, with the lowest being the latest one)
        with open(rotation_file, "a") as w:
            writer = csv.writer(w)
            writer.writerow([picture.picture_name, picture_rotation])


def all_steps_manual(mosaic_name, export_cropped=None, export_rotated=None, show_cropping=None):
    mosaic = Mosaic(dir="to_treat", mosaic_name=mosaic_name)
    # coords = []
    # print(f"coords : {coords}")
    contours, img = contour_manual(mosaic)
    crop_mosaic(mosaic, export_cropped=export_cropped, show_image=show_cropping)
    for i in range(mosaic.num_contours_final):
        picture_name = mosaic.cropped_pictures["filename"][i]
        cv2_array = mosaic.cropped_pictures["img"][i]
        picture = Picture(picture_name=picture_name, cv2_array=cv2_array)
        rotate_manual(picture, save_pic=True)


def all_steps_manual_multiple(num_pictures: int = None, start_index=0):
    all_pictures = [file for file in sorted(os.listdir(TO_TREAT_DIR)) if (file.endswith(".jpg") or file.endswith(".png"))]
    if num_pictures is not None:
        final_index = len(all_pictures) - 1 if start_index + num_pictures > len(all_pictures) else start_index + num_pictures - 1
        files_to_process = all_pictures[start_index:final_index]
    else:
        files_to_process = all_pictures[start_index:]
    for filename in files_to_process:
        print(f"Treating {filename}")
        all_steps_manual(filename, export_cropped=True, export_rotated=True, show_cropping=True)


def rotate_manual_multiple(num_pictures: int = None, start_index=0, dir=CROPPED_DIR, save_pic=None):
    """
    # Already done until 49. Will do 20 more
    # To know where to start : go to rotation_metadata.csv : row_number = 5 :
    #  - means you've rotated 4 pictures (because of headers in the cvs)
    #  - So you've already processed the list of pictures until index = 3
    #  - Hence you can have a start_index = 4, so that you don't process again the 4th picture (at index 3)
    #  - start_index = n-1 where n = # of rows in the rotation_metadata.csv

    # Example : 96 rows. Last picture processed was mamie0038_01.jpg
    # Should start index at 95
    # rotate_manual_multiple(num_pictures=30, start_index=95, save_pic=None)
    # Will start at mamie0038_02.jpg

    # mamie0056_02.jpg at row = 125 in rotation_metadata.csv
    # Should start at index = 124
    """
    all_pictures = [file for file in sorted(os.listdir(CROPPED_DIR)) if (file.endswith(".jpg") or file.endswith(".png"))]
    if num_pictures is not None:
        final_index = len(all_pictures) - 1 if start_index + num_pictures > len(all_pictures) else start_index + num_pictures - 1
        files_to_process = all_pictures[start_index:final_index]
    else:
        files_to_process = all_pictures[start_index:]
    for filename in files_to_process:
        picture = Picture(filename)
        rotate_manual(picture, save_pic=save_pic)


if __name__ == "__main__":

    # all_steps_manual_multiple(start_index=6)

    # Taking all pictures rotated automically, and reputting them in the right orientation (in the ROTATED_MANUAL directory)
    rotate_manual_multiple(dir=ROTATED_AUTO_DIR, save_pic=True)

    # mosaic = Mosaic(dir="to_treat", "mamie0289.jpg")
    # coords = []
    # print(f"coords : {coords}")
    # all_steps_manual("mamie0289.jpg", export_cropped=True, export_rotated=True, show_cropping=True)

    # rotate_manual_multiple(num_pictures=50, start_index=124, save_pic=None)
