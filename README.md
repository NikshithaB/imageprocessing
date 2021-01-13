# imageprocessing
Program1:Program to display grayscale image
Description:
     Grayscale: A grayscale (or graylevel) image is simply one in which the only colors are shades of gray. ... Often, the grayscale intensity is stored as an 8-bit integer giving 256 possible different shades of gray from black to white.
     Binary image:A binary image is one that consists of pixels that can have one of exactly two colors, usually black and white. ... In Photoshop parlance, a binary image is the same as an image in "Bitmap" mode. Binary images often arise in digital image processing as masks or thresholding, and dithering.
To read an image we use cv2.imread() function
to write an image we use cv2.imwrite() function
Program:
import numpy as np
import cv2
image=cv2.imread('flower1.jpg',1)
cv2.imshow('Original', image) 
cv2.waitKey(0)
cv2.destroyAllWindows() 
cv2.imwrite("grayscale.png",image) 
![image](https://user-images.githubusercontent.com/72548737/104420307-84cfce80-552e-11eb-9c2c-f3820f433f9c.png)
![image](https://user-images.githubusercontent.com/72548737/104423509-0c1f4100-5533-11eb-9072-8f90ab7293c6.png)

