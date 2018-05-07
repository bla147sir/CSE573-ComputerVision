import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy import signal

#Load an color image in grayscale
img = cv2.imread('/Users/chenshihchia/Desktop/CSE 573 CV/homework/hw4/MixedVegetables.jpg', 0)
print img
#print type(img) #get the type of img

R = img.shape[0]    #row
C = img.shape[1]    #column

img2 = [[0 for x in range(2*C-1)] for y in range(2*R-1)]
img2 = np.asarray(img2) 
#print img2.shape[0]
#print img2.shape[1]
k = 0
h = 0

for i in range(0,2*R,2):
    for j in range(0,2*C,2):
        img2[i][j] = img[k][h]
        h = h+1
    k = k+1
    h = 0

k = 0
h = 0
for i in range(0,R,1):
    for j in range(1,C,1):
        if img[k][h]>img[k][h+1] :
            img2[2*i][2*j-1] = img[k][h] - img[k][h+1]
        else :
            img2[2*i][2*j-1] = img[k][h+1] - img[k][h]
        h = h+1
    h = 0
    k = k+1

k = 0
h = 0
for i in range(1,2*R-1,2):
    for j in range(0,2*C,2):
        if img[k][h] > img[k+1][h] :
            img2[i][j] = img[k][h] - img[k+1][h]
        else:
            img2[i][j] = img[k+1][h] - img[k][h]
        h = h+1
    h = 0
    k = k+1
        
print img2


#show images
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img2, cmap = 'gray')
plt.title('crack edge Image'), plt.xticks([]), plt.yticks([])
plt.show()