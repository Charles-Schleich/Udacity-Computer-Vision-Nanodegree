# Dialation and Erosion:


```python
# Reads in a binary image
image = cv2.imread(‘j.png’, 0) 

# Create a 5x5 kernel of ones
kernel = np.ones((5,5),np.uint8)

# Dilate the image
dilation = cv2.dilate(image, kernel, iterations = 1)
```


```python
# Erode the image
erosion = cv2.erode(image, kernel, iterations = 1)
```

# Openning 
Opening: which is erosion followed by dilation
Useful in noise reduction: Erosion first gets rid of noise, shrinks the object and grows it again.
Noise dies out but the object survives 
(effectively getting rid of noise outside the object)

```python
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
```

# Closing
Closing: which is dialation followed by erosion
Useful for for closing small holes or dark areas inside an object, 
Holes in the object die out but the object survives
(effectively getting rid of noise INSIDE the object)

```python
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
```
