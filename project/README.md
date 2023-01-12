# README.md

## Process that seems to be fairly good

- Before Canny Edge : there needs to be a thresholding part for sure
  - So far : thresholding :
  ```python
  th0 = cv2.adaptiveThreshold(
      img_grey,
      255,
      cv2.ADAPTIVE_THRESH_MEAN_C,
      cv2.THRESH_BINARY,
      blockSize=3,
      C=2,
  )
  ```
seems to do a good job
  - Combined with Canny Edge : 
  ```python
  c0 = cv2.Canny(thresh, threshold1=100, threshold2=200, L2gradient=True)
  ```

## Canny edges 
- No need 

