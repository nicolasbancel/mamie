# Rotation Documentation

## Links

- [Deep Learning for image rotation](https://medium.com/analytics-vidhya/how-to-auto-rotate-the-image-using-deep-learning-c34b2e0e157d)

## Choices

- Gave up with Haar Cascade : way too many false positives. Works well for configuration of a given image, but does not scale when running on a big chunk of pictures (since parameters would need to be re adjusted often)


## Corner cases - Difficult transformations

- `"mamie0007_02.jpg"` : fixed with improvement to 2nd model (2nd highest confidence score after 1.00)
- `"mamie0014_01.jpg"` : fixed with improvement to 2nd model (2nd highest confidence score after 1.00)
- `"mamie0000_03.jpg", "mamie0001_04.jpg", "mamie0011_01.jpg"` : some fixed with the density function, some not.
- Corner cases : run the following in ipython

```python

from rotate import *

tough_ones = ["mamie0003_01.jpg",
"mamie0000_03.jpg",
"mamie0001_04.jpg",
"mamie0004_02.jpg",
"mamie0007_01.jpg",
"mamie0011_01.jpg",
"mamie0013_04.jpg",
"mamie0014_01.jpg",
"mamie0036_03.jpg",
"mamie0038_01.jpg",
"mamie0039_01.jpg",
"mamie0039_03.jpg",
"mamie0010_01.jpg",
"mamie0010_02.jpg",
"mamie0010_03.jpg",
"mamie0004_02.jpg"
]

rotate_all(picture_list=tough_ones, log=True, show_steps=True):

```

## Scripts 

- Run 

# Residual black triangle

- Not done well for
  - `"mamie0022_02.jpg"`
  - `"mamie0028_02.jpg"`
  - `"mamie0038_01.jpg"`
  - `"mamie0057_02.jpg"`
- Example of black residual triangle that is detected in the contour (with no impact on final shape):
  - `mamie0101.jpg`
- Example of black residual triangle that is detected in the contour (with impact on final shape):
  - `mamie0098.jpg` - and more precisely `mamie0098_01` (landeau is cropped)
  - `mamie0140.jpg` - and more precisely enventually on Cyril's picture : `mamie0140_02.jpg` ✅
  - `mamie0138.jpg` - and more precisely Maman au jardin : `mamie0138_01.jpg` ✅
  - `mamie0276.jpg` - and more precisely Papy ski nautique : `mamie0276_02.jpg`✅
  - `mamie0261.jpg` - and more precisely Papa retour Boston : `mamie0261_03.jpg` ✅


["mamie0140.jpg", "mamie0138.jpg", "mamie0276.jpg","mamie0261.jpg"]


See the black edge entirely : 
```python
from Mosaic import *

mosaic = Mosaic(mosaic_name = "mamie0171.jpg")


mosaic = Mosaic(mosaic_name = "mamie0022.jpg")
show("Img Original", mosaic.img)
grey = cv2.cvtColor(mosaic.img, cv2.COLOR_BGR2GRAY)
show("Img Grey", grey)
```


["mamie0003.jpg","mamie0000.jpg","mamie0001.jpg","mamie0004.jpg","mamie0007.jpg","mamie0011.jpg","mamie0013.jpg","mamie0014.jpg","mamie0036.jpg"]

# Residual black edge

- Run without removing the bad point, and exporting all contours : `mamie0008.jpg` should have an issue 

- Small residual black edge, not a bad point, but creates a weird angle / margin in the contour
  - `mamie0024.jpg`
  - `mamie0030.jpg` 


- It makes a couple of mosaics fail
  - `mamie0184.jpg`
  - `mamie0185.jpg`
  - `mamie0186.jpg`
  - `mamie0188.jpg`
  - `mamie0193.jpg`
  - `mamie0210.jpg` (to be precise : `mamie0210_04.jpg`)
  - `mamie0211.jpg`
  - Run `mamie0210.jpg` with correct config : `python3 main.py -m "mamie0210.jpg" --no-log_contouring --no-log_rotations -exco "all" -excr -exro --show_contouring --show_cropping --no-show_rotation`
- Run first 20 mosaics : `python3 main.py -n 20 --no-log_contouring --no-log_rotations -exco "all" -excr -exro --no-show_contouring --no-show_cropping --no-show_rotation`
  - After bad point removal : 
    - `python3 main.py -m "mamie0184.jpg" "mamie0185.jpg" "mamie0186.jpg" "mamie0193.jpg" "mamie0210.jpg" "mamie0211.jpg" --no-log_contouring --no-log_rotations -exco "all" -excr -exro --show_contouring --show_cropping --no-show_rotation`
  - Fixed after bad point removal = True ✅
