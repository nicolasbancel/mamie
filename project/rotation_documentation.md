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
