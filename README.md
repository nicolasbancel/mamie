# mamie

## Description

Script which automatically crops and rotates pictures. There are 3 main "objects" : 
- a Mosaic - which is a scan of multiple pictures all scanned at once. The source data is made of mosaics. 
- a Contour - the script detects the main contours in the mosaic (eventually trying to narrow the contours down to the exact contours / sides of the pictures in the mosaic)
- a Picture - which is the inside of a contour, cropped / extracted from the mosaic, remodeled into a rectangle shape, and rotated.

## Project structure
- `data/mosaic/source` : mosaics from the scanning you've done
- `project/pictures_per_mosaic.csv` : Metadata file, which logs how many pictures are supposed to be found in the mosaic. This is used to determine whether the automatic contouring is successful
- `project/rotation_metadata.csv` : Metadata file, which logs, based on the cropped pictures, by how many 90° rotations they should be rotated. The metadata is generated by running the `rotate_manual.py` file. Which shows pictures, asks you to validate the correct rotation, and it automatically writes into the `.csv` file.
- Data folders
  - `data/contoured/failure` : when in a mosaic, the number of pictures detected differ from the true number of pictures (referenced in `pictures_per_mosaic.csv`), or when one of the contours has strictly more than 5 corners (which means there are some issues with the contour : like an outlier point). 5 is accepted because it can come from a scission that has been done (explanations later). The failing mosaic is copied into `data/to_treat_manually`, where you'll have to do the cropping yourself.
  -  `data/contoured/success` : log of the contours done well. Steps stored in a .jpg file
  - `data/cropped` : the contours are then extracted from the original picture. Since the contour may not be 100% straight, it is reshaped / rotated into a straight rectangle using the warpAffine method (not Perspective because the angles are already 90% : there's no need to try to extrapolate what a front view would look like : it is already a front view, with 90° angles : they are just rotated)
  - `data/rotated_automatic` : cropped pictures are then rotated. Method : trying 4 rotations of 90°, determining the rotation where the highest / cleanest number of faces are detected. Order of priority :
    - # of faces detected with accuracy = 1
    - 2nd highest accuracy score after 1
    - Highest area of faces captured above the middle line, weighted by the accuracy (this assumes in most pictures with faces that the faces are above the middle - which is not really correct - it doesn't work all the time)
    - 2 last methods of ranking are useless and deprecated
      - lowest density of points (eyes, nose, tips of mouth) : it seemed like well identified faces have a spread out distribution of points. Hence a lower density
      - highest identified area of faces (weighted by the accuracy)
    - This method gives a 90% accuracy on rotations : good, considering it is not able to rotate the landscapes, since there's no face on those. Hence the % is even higher when there are faces on the picture.
- `project/results/results_contours.csv` : logs the configuration used for the run, and all the information related to how accurate the contouring was (# contours found, areas of main contours, # points per contour, whether the # contours found corresponds to the # of pictures). We're at a 89% accuracy per mosaic
- `project/results/results_rotations.csv` : Per picture, per run, logs what 90° rotation was picked, whether it matches the correct rotation (if correct one has been logged in `project/rotation_metadata.csv`), and the info captured about each 90° rotation, to retrace the decision.

## Process and commands 

- Need to install `poetry` and run `poetry install` to install all the libraries
- Drop your mosaics into the `data/mosaic/source` folder
- Typical command to run :
  - `python3 main.py -log_c -log_r -exco "fail_only" -excr -exro --no-show_contouring --no-show_cropping --no-show_rotation`
    - This will run through all the mosaics in `source`
    - `-log_c` : log the results of the contour accuracy in `results_contours.csv`
    - `-log_r` : log the results of the rotation accuracy in `results_rotations.csv`
    - `-exco "fail_only"` : exports the contour summary only when the contouring in the mosaic fails. Exports to `data/contoured/failure`
    - `-excr` : exports the cropped + warpAffine pictures to `data/cropped`
    - `-exro` : exports the rotated picture to `data/rotated_automatic`
    - `--no-show_contouring` : does not show the steps of the contouring
    - `--no-show_cropping` : does not show the steps of the cropping the pictures
    - `--no-show_rotation` : does not show the steps of the rotation of the pictures
  - Possible argument : 
    - `-n 20` : runs the 20 first mosaics of the `data/mosaic/source` folder
    - `-m "mamie0003.jpg" "mamie0000.jpg" "mamie0001.jpg"` : runs all the steps only for this list of 3 mosaics.


## Examples
- See the mosaics and pictures already dropped in the `data` folders.
- For 1 specific example : 
  - [Step one](https://github.com/nicolasbancel/mamie/tree/main/data/mosaic/all_steps/01_mamie0009_mosaic.jpeg) : Treating `mamie0009.jpg` mosaic first
  - [Step two](https://github.com/nicolasbancel/mamie/tree/main/data/mosaic/all_steps/02_mamie0009_mosaic_contoured.jpeg) : Contour is successful. 3 main contours identified (3 pictures are supposed to be found) & contours have no more than 5 corners. The picture at the bottom is not straight, it has a rectangle
  - [Step three](https://github.com/nicolasbancel/mamie/tree/main/data/mosaic/all_steps/03_mamie0009_03_boundingrectangle.jpeg) : cropping can only be done on a "rectangle", which has a top left corner, and a bottom right corner. It is not possible to extract from a numpy array an "angled" rectangle. What's extracted is the bounding rectangle, with its rotation + center of rotation captured, though. If no other modification was made, we would end up with the black edges seen in this picture;
  - [Step four](https://github.com/nicolasbancel/mamie/tree/main/data/mosaic/all_steps/04_mamie0009_03_warpaffine.jpeg) : using the metadata available for that rectangle, we rotate it with the warpAffine method.
  -[Step five](https://github.com/nicolasbancel/mamie/tree/main/data/mosaic/all_steps/04_mamie0009_03_rotated.jpeg) : rotation is done using the method explained above, in section **Project structure**

