# imageprocessing
**Program1:Program to display grayscale image**
Description:
     Grayscale: A grayscale (or graylevel) image is simply one in which the only colors are shades of gray. ... Often, the grayscale intensity is stored as an 8-bit integer giving 256 possible different shades of gray from black to white.
     Binary image:A binary image is one that consists of pixels that can have one of exactly two colors, usually black and white. ... In Photoshop parlance, a binary image is the same as an image in "Bitmap" mode. Binary images often arise in digital image processing as masks or thresholding, and dithering.
To read an image we use cv2.imread() function
to write an image we use cv2.imwrite() function
cv2. waitKey() is a keyboard binding function. ... The function waits for specified milliseconds for any keyboard event.
cv2. destroyAllWindows() simply destroys all the windows we created

Program:
```python
import numpy as np
import cv2
image=cv2.imread('flower1.jpg',1)
cv2.imshow('Original', image) 
cv2.waitKey(0)
cv2.destroyAllWindows() 
cv2.imwrite("grayscale.png",image) 
```
![image](https://user-images.githubusercontent.com/72548737/104420307-84cfce80-552e-11eb-9c2c-f3820f433f9c.png)
![image](https://user-images.githubusercontent.com/72548737/104423509-0c1f4100-5533-11eb-9072-8f90ab7293c6.png)


Program2:Program to performe linear transformation

Description:
Linear Transformation:Piece-wise Linear Transformation is type of gray level transformation that is used for image enhancement. It is a spatial domain method. It is used for manipulation of an image so that the result is more suitable than the original for a specific application.

Resizing an image means changing the dimensions of it, be it width alone, height alone or changing both of them. Also, the aspect ratio of the original image could be preserved in the resized image. To resize an image, OpenCV provides cv2.resize() function.
Rotation: This is a simple example of a linear transformation. Linear Transformations A transformation of the plane is called a linear transformation if it corresponds to multiplying each point (x, y) by some 2 × 2 matrix A, i.e. ... Rotation of the plane by any angle around the origin.
Program:
```python
import cv2
image= cv2.imread('flower1.jpg')
scale_percent = 500
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dsize = (width, height)
output = cv2.resize(image, dsize)
cv2.imshow('Original',output) 
cv2.waitKey(0)
```
Output:


![image](https://user-images.githubusercontent.com/72548737/104424351-1e4daf00-5534-11eb-87df-36464688837a.png)


Program2(b): Rotation
```python
import cv2
image=cv2.imread('flower1.jpg')
cv2.imshow('original',image)
src=cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)
cv2.imshow('output',src)
cv2.waitKey(0)
```
Output:
![image](https://user-images.githubusercontent.com/72548737/104424839-bc417980-5534-11eb-9d19-c5ed866b2039.png)



Program3:Program to find sum and mean of the image
Description:
Mean:'mean' value gives the contribution of individual pixel intensity for the entire image & variance is normally used to find how each pixel varies from the neighbouring pixel (or centre pixel) and is used in classify into different regions.
sum:Adding Images To add two images or add a constant value to an image. • [imadd] function adds the value of each pixel in one of the input images with the corresponding pixel in the other input image and returns the sum in the corresponding pixel of the output image
A = imread( filename ) reads the image from the file specified by filename , inferring the format of the file from its contents.
 cv2. imread() method loads an image from the specified file. 
 program:
 ```python
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
```
output:
![image](https://user-images.githubusercontent.com/72548737/104435362-252eee80-5541-11eb-93f9-2d28d3f00786.png)
![image](https://user-images.githubusercontent.com/72548737/104435489-50194280-5541-11eb-8120-fd7171a0e09d.png)




program4:Convert the image to gray scale and binary image
description:
Grayscale: A grayscale (or graylevel) image is simply one in which the only colors are shades of gray. ... Often, the grayscale intensity is stored as an 8-bit integer giving 256 possible different shades of gray from black to white.
     Binary image:A binary image is one that consists of pixels that can have one of exactly two colors, usually black and white. ... In Photoshop parlance, a binary image is the same as an image in "Bitmap" mode. Binary images often arise in digital image processing as masks or thresholding, and dithering.
To read an image we use cv2.imread() function
Threshold:Thresholding is a technique in OpenCV, which is the assignment of pixel values in relation to the threshold value provided. In thresholding, each pixel value is compared with the threshold value. If the pixel value is smaller than the threshold, it is set to 0, otherwise, it is set to a maximum value (generally 255).
program:
```python
import cv2
img = cv2.imread('flower1.jpg')
cv2.imwrite('graynature.jpg',img)
cv2.imshow('Original',img,)
img = cv2.imread('flower1.jpg',0)
cv2.imwrite('gray.jpg',img)
cv2.imshow('Origi',img,)
img = cv2.imread('flower1.jpg', 2) 
ret, bw_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY) 
cv2.imshow("Binary", bw_img) 
cv2.waitKey(0)
cv2.destroyAllWindows()
```

output:

![image](https://user-images.githubusercontent.com/72548737/104429403-6c65b100-553a-11eb-95cc-e31d22376c7c.png)
![image](https://user-images.githubusercontent.com/72548737/104429595-a636b780-553a-11eb-927f-a98d6f5422d0.png)
![image](https://user-images.githubusercontent.com/72548737/104429708-c8303a00-553a-11eb-844c-884e8eaec05e.png)



Program 5:Program to covert the given image to different color space
description:

BGR2RGB:When the image file is read with the OpenCV function imread() , the order of colors is BGR (blue, green, red). On the other hand, in Pillow, the order of colors is assumed to be RGB (red, green, blue). Therefore, if you want to use both the Pillow function and the OpenCV function, you need to convert BGR and RGB.
BGR2HSV:HSV color space is the most suitable color space for color based image segmentation. ... In OpenCV, value range for 'hue', 'saturation' and 'value' are respectively 0-179, 0-255 and 0-255.
Ycrcb: YCbCr represents colors as combinations of a brightness signal and two chroma signals.

program:
```python
import cv2 
img = cv2.imread('flower1.jpg') 
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
img2= cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img3=cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)   
cv2.imshow('image', img1)
cv2.waitKey(0)
cv2.imshow('image', img2) 
cv2.waitKey(0)
cv2.imshow('image', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
output:
![image](https://user-images.githubusercontent.com/72548737/104431785-29f1a380-553d-11eb-91d9-ef122bab015b.png)
![image](https://user-images.githubusercontent.com/72548737/104431936-560d2480-553d-11eb-8386-a9ddc44d623f.png)
![image](https://user-images.githubusercontent.com/72548737/104432133-8e146780-553d-11eb-9e5f-fb9a80d4cbbe.png)


Program6:Create an image from 2d array

Description:
     Python Imaging Library (abbreviated as PIL) (in newer versions known as Pillow) is a free and open-source additional library for the Python programming language that adds support for opening, manipulating, and saving many different image file formats
To make a numpy array, you can just use the np. array() function. All you need to do is pass a list to it, and optionally, you can also specify the data type of the data     
program:
```python
import numpy as np
from PIL import Image
import cv2
array = np.zeros([100, 200, 3], dtype=np.uint8)
array[:,:100] = [200, 200, 200] 
array[:,100:] = [200, 100, 200]   

img = Image.fromarray(array)
img.save('testrgb.png')
img.show()
cv2.waitKey(0)
```
output:
![image](https://user-images.githubusercontent.com/72548737/104433816-5e665f00-553f-11eb-92c7-87ee39de9131.png)


program7:Program to find the neighbourhood of matrix

description:
 Given a  matrix and a set of cell indexes e.g., an array of (i, j) where i indicates row and j column. For every given cell index (i, j), finding sums of all matrix elements except the elements present in i’th row and/or j’th column.
 
The function "shape" returns the shape of an array. The shape is a tuple of integers. These numbers denote the lengths of the corresponding array dimension
import numpy as np
```python

M = [[1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]]

M = np.asarray(M)
N = np.zeros(M.shape)
def sumNeighbors(M,x,y):
    l = []
    for i in range(max(0,x-1),x+2): # max(0,x-1), such that no negative values in range()
        for j in range(max(0,y-1),y+2):
            try:
                t = M[i][j]
                l.append(t)
            except IndexError: # if entry doesn't exist
                pass
    return sum(l)-M[x][y] # exclude the entry itself
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        N[i][j] = sumNeighbors(M, i, j)

print ("Original matrix:\n", M)
print ("Summed neighbors matrix:\n", N)
```
Output:
Original matrix:
 [[1 2 3]
 [4 5 6]
 [7 8 9]]
Summed neighbors matrix:
 [[11. 19. 13.]
 [23. 40. 27.]
 [17. 31. 19.]]

**program8: write c++ program to perform Operator overloading**
description:
Using operator overloading in C++, you can specify more than one meaning for an operator in one scope. The purpose of operator overloading is to provide a special meaning of an operator for a user-defined data type. With the help of operator overloading, you can redefine the majority of the C++ operators.
program:
```python
#include <iostream>
using namespace std;
class matrix
{
 int r1, c1, i, j, a1;
 int a[10][10];

public:int get()
 {
  cout << "Enter the row and column size for the  matrix\n";
  cin >> r1;
  cin >> c1;
   cout << "Enter the elements of the matrix\n";
  for (i = 0; i < r1; i++)
  {
   for (j = 0; j < c1; j++)
   {
    cin>>a[i][j];

   }
  }
 
 
 };
 void operator+(matrix a1)
 {
 int c[i][j];
  
   for (i = 0; i < r1; i++)
   {
    for (j = 0; j < c1; j++)
    {
     c[i][j] = a[i][j] + a1.a[i][j];
    }
   
  }
  cout<<"addition is\n";
  for(i=0;i<r1;i++)
  {
   cout<<" ";
   for (j = 0; j < c1; j++)
   {
    cout<<c[i][j]<<"\t";
   }
   cout<<"\n";
  }

 };

  void operator-(matrix a2)
 {
 int c[i][j];
  
   for (i = 0; i < r1; i++)
   {
    for (j = 0; j < c1; j++)
    {
     c[i][j] = a[i][j] - a2.a[i][j];
    }
   
  }
  cout<<"subtraction is\n";
  for(i=0;i<r1;i++)
  {
   cout<<" ";
   for (j = 0; j < c1; j++)
   {
    cout<<c[i][j]<<"\t";
   }
   cout<<"\n";
  }
 };

 void operator*(matrix a3)
 {
  int c[i][j];

  for (i = 0; i < r1; i++)
  {
   for (j = 0; j < c1; j++)
   {
    c[i][j] =0;
    for (int k = 0; k < r1; k++)
    {
     c[i][j] += a[i][k] * (a3.a[k][j]);
    }
  }
  }
  cout << "multiplication is\n";
  for (i = 0; i < r1; i++)
  {
   cout << " ";
   for (j = 0; j < c1; j++)
   {
    cout << c[i][j] << "\t";
   }
   cout << "\n";
  }
   };

};

int main()
{
 matrix p,q;
 p.get();
 q.get();
 p + q;
 p - q;
 p * q;
return 0;
}
```
```
  **output:**
  Enter the row and colunm size for the matrix
  2
  2
  Enter the element of the matrix
  6
  7
  5
  8
  Enter the row and colunm size for the matrix
  2
  2
  Enter the element of the matrix
  2
  3
  1
  4
  addition is
  8 10
  6
  12
  subtraction is
  4 4
  4 4
  multiplication is
  19 46
  18
  47
```
**program 9:develop a program to find the neighbor of each elememt of matrix**
import numpy as np
```python
i=0
j=0
a= np.array([[1,2,3,4,5], [2,3,4,5,6],[3,4,5,6,7],[4,5,6,7,8],[5,6,7,8,9]])
print("a : ",str(a))
def neighbors(radius, rowNumber, columnNumber):
     return [[a[i][j] if  i >= 0 and i < len(a) and j >= 0 and j < len(a[0]) else 0
                for j in range(columnNumber-1-radius, columnNumber+radius)]
                    for i in range(rowNumber-1-radius, rowNumber+radius)]
neighbors(1, 2, 3)
```
**output**
```
a :  [[1 2 3 4 5]
 [2 3 4 5 6]
 [3 4 5 6 7]
 [4 5 6 7 8]
 [5 6 7 8 9]]
[[2, 3, 4], [3, 4, 5], [4, 5, 6]]
```


**program 10:Develop a program to implement negative transformation**
description:
The second linear transformation is negative transformation, which is invert of identity transformation. In negative transformation, each value of the input image is subtracted from the L-1 and mapped onto the output image
**program:**
```python
import cv2
img=cv2.imread("p3.jpg")
img_neg = 255-img
cv2.imshow('Original',img)  
cv2.waitKey(0)
cv2.imshow('negation',img_neg) 
cv2.waitKey(0)
cv2.destroyAllWindows()
```
Output:

