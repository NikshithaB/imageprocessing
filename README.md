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


Program2:Program to performe linear transformation

Description:
Linear Transformation:Piece-wise Linear Transformation is type of gray level transformation that is used for image enhancement. It is a spatial domain method. It is used for manipulation of an image so that the result is more suitable than the original for a specific application.

Resizing an image means changing the dimensions of it, be it width alone, height alone or changing both of them. Also, the aspect ratio of the original image could be preserved in the resized image. To resize an image, OpenCV provides cv2.resize() function.
Rotation: This is a simple example of a linear transformation. Linear Transformations A transformation of the plane is called a linear transformation if it corresponds to multiplying each point (x, y) by some 2 × 2 matrix A, i.e. ... Rotation of the plane by any angle around the origin.
Program:
import cv2
image= cv2.imread('flower1.jpg')
scale_percent = 500
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dsize = (width, height)
output = cv2.resize(image, dsize)
cv2.imshow('Original',output) 
cv2.waitKey(0)
Output:
![image](https://user-images.githubusercontent.com/72548737/104424351-1e4daf00-5534-11eb-87df-36464688837a.png)


Program2(b): Rotation
import cv2
image=cv2.imread('flower1.jpg')
cv2.imshow('original',image)
src=cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)
cv2.imshow('output',src)
cv2.waitKey(0)

Output:
![image](https://user-images.githubusercontent.com/72548737/104424839-bc417980-5534-11eb-9d19-c5ed866b2039.png)



Program3:Program to find sum and mean of the image
Description:
Mean:'mean' value gives the contribution of individual pixel intensity for the entire image & variance is normally used to find how each pixel varies from the neighbouring pixel (or centre pixel) and is used in classify into different regions.
sum:Adding Images To add two images or add a constant value to an image. • [imadd] function adds the value of each pixel in one of the input images with the corresponding pixel in the other input image and returns the sum in the corresponding pixel of the output image
A = imread( filename ) reads the image from the file specified by filename , inferring the format of the file from its contents.
 cv2. imread() method loads an image from the specified file. 
 program:
import cv2
import os
path = 'C:\Pictures'
imgs = []

files = os.listdir(path)
for file in files:
    filepath=path+"\\"+file
    imgs.append(cv2.imread(filepath))
i=0
im = []
for im in imgs:
    #cv2.imshow(files[i],imgs[i])
    im+=imgs[i]
    i=i+1
cv2.imshow("sum of four pictures",im)
meanImg = im/len(files)
cv2.imshow("mean of four pictures",meanImg)
cv2.waitKey(0)

output:
![image](https://user-images.githubusercontent.com/72548737/104429403-6c65b100-553a-11eb-95cc-e31d22376c7c.png)
![image](https://user-images.githubusercontent.com/72548737/104429595-a636b780-553a-11eb-927f-a98d6f5422d0.png)

