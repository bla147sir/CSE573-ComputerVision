import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy import signal

#Load an color image in grayscale
img = cv2.imread('/Users/chenshihchia/Desktop/CSE 573 CV/homework/hw4/UBCampus.jpg', 0)
#print img
#print type(img) #get the type of img

C = img.shape[0]    #column
R = img.shape[1]    #row


# DoG convolution
img_DoG = [[0 for x in range(R)] for y in range(C)]  #img after DoG convolution
dogk = [ [ 0, 0,-1,-1,-1, 0, 0],
         [ 0,-2,-3,-3,-3,-2, 0],
         [-1,-3, 5, 5, 5,-3,-1],
         [-1,-3, 5,16, 5,-3,-1],
         [-1,-3, 5, 5, 5,-3,-1],
         [ 0,-2,-3,-3,-3,-2, 0],
         [ 0, 0,-1,-1,-1, 0, 0]  ] #DoG's mask or fliter

img_DoG = signal.convolve2d(img, dogk, boundary='symm', mode='same')
#print img_DoG

#zero-crossing of DoG
img_zero = [[255 for x in range(R)] for y in range(C)] #img after zero crossing
for i in range(1,C):
    for j in range(1,R):
        if (img_DoG[i][j]*img_DoG[i][j-1]<0 or img_DoG[i][j]*img_DoG[i-1][j]<0):
            img_zero[i][j] = 0            
#print img_zero

#Sobel find the first derivative, use to remove the weak edge
h1 = [[ 1, 2, 1],
      [ 0, 0, 0],
      [-1,-2,-1]  ]

h2 = [[ 0,-1,-2],
      [ 1, 0,-1],
      [ 2, 1, 0]  ]

h3 = [[-1, 0, 1],
      [-2, 0, 2],
      [-1, 0, 1]  ]

h4 = [[-2,-1, 0],
      [-1, 0, 1],
      [ 0, 1, 2]  ]
      
h1 = np.asarray(h1)
h2 = np.asarray(h2)
h3 = np.asarray(h3)
h4 = np.asarray(h4)
h5 = -h1
h6 = -h2
h7 = -h3
h8 = -h4

img_sob1 = signal.convolve2d(img, h1, boundary='symm', mode='same') #img after sobel h1
img_sob2 = signal.convolve2d(img, h2, boundary='symm', mode='same') #img after sobel h2
img_sob3 = signal.convolve2d(img, h3, boundary='symm', mode='same') #img after sobel h3
img_sob4 = signal.convolve2d(img, h4, boundary='symm', mode='same') #img after sobel h4
img_sob5 = signal.convolve2d(img, h5, boundary='symm', mode='same') #img after sobel h5
img_sob6 = signal.convolve2d(img, h6, boundary='symm', mode='same') #img after sobel h6
img_sob7 = signal.convolve2d(img, h7, boundary='symm', mode='same') #img after sobel h7
img_sob8 = signal.convolve2d(img, h8, boundary='symm', mode='same') #img after sobel h8

#print img_sob1
threshold = 80

img_zero_s = [[255 for x in range(R)] for y in range(C)]

for i in range(C):
    for j in range(R):
        img_zero_s[i][j] = img_zero[i][j]
        
for i in range(C):
    for j in range(R):
        if (img_zero_s[i][j]==0 and img_sob1[i][j]<threshold 
            and img_sob2[i][j]<threshold and img_sob3[i][j]<threshold 
            and img_sob4[i][j]<threshold and img_sob5[i][j]<threshold 
            and img_sob6[i][j]<threshold and img_sob7[i][j]<threshold and img_sob8[i][j]<threshold):
            img_zero_s[i][j] = 255
                            
#LoG convolution
img_LoG = [[0 for x in range(R)] for y in range(C)]  #img after LoG convolution
Logk = [ [ 0, 0, 1, 0, 0],
         [ 0, 1, 2, 1, 0],
         [ 1, 2,-16,1, 2],
         [ 0, 0, 1, 0, 0],
         [ 0, 1, 2, 1, 0]  ] #LoG's mask or fliter

img_LoG = signal.convolve2d(img, Logk, boundary='symm', mode='same')

#zero-crossing of LoG
img_zero_LoG = [[255 for x in range(R)] for y in range(C)] #img after zero crossing
for i in range(1,C):
    for j in range(1,R):
        if (img_LoG[i][j]*img_LoG[i][j-1]<0 or img_LoG[i][j]*img_LoG[i-1][j]<0):
            img_zero_LoG[i][j] = 0  

#remove the weak edges
img_zero_LoG_s = [[255 for x in range(R)] for y in range(C)]

for i in range(C):
    for j in range(R):
        img_zero_LoG_s[i][j] = img_zero_LoG[i][j]
        
for i in range(C):
    for j in range(R):
        if (img_zero_LoG_s[i][j]==0 and img_sob1[i][j]<threshold 
            and img_sob2[i][j]<threshold and img_sob3[i][j]<threshold 
            and img_sob4[i][j]<threshold and img_sob5[i][j]<threshold 
            and img_sob6[i][j]<threshold and img_sob7[i][j]<threshold and img_sob8[i][j]<threshold):
            img_zero_LoG_s[i][j] = 255
                            

#show images
plt.subplot(231),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(232),plt.imshow(img_DoG, cmap = 'gray')
plt.title('DoG Image'), plt.xticks([]), plt.yticks([])
plt.subplot(233),plt.imshow(img_zero,cmap = 'gray')
plt.title('Zero-crossing of DoG'), plt.xticks([]), plt.yticks([])
plt.subplot(234),plt.imshow(img_zero_s,cmap = 'gray')
plt.title('DoG with Strong edges'), plt.xticks([]), plt.yticks([])
plt.subplot(235),plt.imshow(img_zero_LoG_s,cmap = 'gray')
plt.title('LoG with Strong edges'), plt.xticks([]), plt.yticks([])
plt.show()