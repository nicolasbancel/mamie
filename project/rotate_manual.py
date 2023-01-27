from Picture import *
import cv2


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


def rotate_manual_multiple(num_pictures: int = None, start_index=0, save_pic=None):
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

    rotate_manual_multiple(num_pictures=50, start_index=124, save_pic=None)
