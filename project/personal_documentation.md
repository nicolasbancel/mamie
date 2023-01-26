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

# Cropping documentation

- Not done well for
  - `"mamie0022_02.jpg"`
  - `"mamie0028_02.jpg"`
  - `"mamie0038_01.jpg"`
  - `"mamie0057_02.jpg"`
- Example of black residual triangle that is detected in the contour (with no impact on final shape):
  - `mamie0101.jpg`
- Example of black residual triangle that is detected in the contour (with impact on final shape):
  - `mamie0098.jpg` - and more precisely `mamie0098_01` (landeau is cropped)
  - `mamie0140.jpg` - and more precisely enventually on Cyril's picture : `mamie0140_02.jpg`
  - `mamie0138.jpg` - and more precisely Maman au jardin : `mamie018_01.jpg`
  - `mamie0276.jpg` - and more precisely Papy ski nautique : `mamie0276_02.jpg`
  - `mamie0261.jpg` - and more precisely Papa retour Boston : `mamie0261_03.jpg`


See the black edge entirely : 
```python
from Mosaic import *

mosaic = Mosaic(mosaic_name = "mamie0171.jpg")
show("Img Original", mosaic.img)
grey = cv2.cvtColor(mosaic.img, cv2.COLOR_BGR2GRAY)
show("Img Grey", grey)
```

