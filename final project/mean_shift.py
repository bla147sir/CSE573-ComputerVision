import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
from random import randint

def mean_shift(F, seed):
    F_copy = F
    seed_copy = seed
    
    #single threshold
    h = 90   
    #spatial threshold
    hs = 100  
    #color threshold
    hr = 100   
    c = 0.75*(hs*hr)**2

    #compute the eucledian distance of all the points with mean
    #indexes of all the points that are within threshold h
    within_h = [[-1 for x in range(5)] for y in range(R*C)]            
    within_h = np.asarray(within_h)    
    
    mean = [0 for x in range(5)]     
    mean = np.asarray(mean)    
    # how many weight eucledian distance is smaller than the threshold
    total_weight = 0   
    for i in range(R*C):
        dis = 0
        dis_s = 0;
        dis_r = 0;
        weight = 0;
        if F_copy[i][5]==-1:
            for j in range(3):
                dis_r = dis_r + (int(seed_copy[j])-int(F_copy[i][j]))**2  
            dis_s = (int(seed_copy[3])-int(F_copy[i][3]))**2 + (int(seed_copy[4])-int(F_copy[i][4]))**2
            dis = dis_s + dis_r
            if dis <= h**2 and dis_s <= hs**2 and dis_r <= hr**2 : 
                weight = (float(c)/float((hs*hr)**2))*(1-float(dis_s)/float((hs**2)))*(1-float(dis_r)/float((hr**2)))   #change c
                total_weight = total_weight + weight
                for j in range(5):
                    within_h[i][j] = weight * F_copy[i][j]

        
    #compute the new mean value
    #initialize
    for i in range(5):
        mean[i] = 0      
    
    for k in range(5):
        for i in range(R*C):
            if within_h[i][k]!=-1 and F_copy[i][5]==-1:
                mean[k] = mean[k] + within_h[i][k]
        mean[k] = mean[k]/total_weight           
    
    
    #check if the mean shift value is below iter
    ite_num = 0     
    #mean shift value
    diff = [0 for x in range(5)]   
    diff = np.asarray(diff)
    #convergence criterion value
    ite = [100 for x in range(5)]     
    ite = np.asarray(ite) 
    
    for i in range(5):
        diff[i] = abs(mean[i] - seed_copy[i])
        if ite[i] > diff[i]:
            ite_num = ite_num + 1
    
    
    #mark the index if mean shift value is below iter        
    if ite_num==5:
        for i in range(R*C):
            if within_h[i][0] != -1:
                F_copy[i][5] = 1
                for j in range(3):
                    F_copy[i][j] = mean[j]    
    
    return F_copy

#load image with color=1
img = cv2.imread('/Users/chenshihchia/Desktop/CSE 573 CV/final project/Image_Hill-House.jpg',1)
#row
R = img.shape[0]    
#column
C = img.shape[1]    

#change the BGR into RGB
for i in range(R):
    for j in range(C):
        temp = img[i][j][0]
        img[i][j][0] = img[i][j][2]
        img[i][j][2] = temp
        
F = [[-1 for x in range(6)] for y in range(C*R)]   
#spatial-range feature matrix    R G B i j mark
#mark the index if mean shift value is below iter   -1, 1
F = np.asarray(F)

#initial seed point = current mean
seed = [0 for x in range(5)]   
seed = np.asarray(seed)

# Build F table
a = 0
b = 0
for i in range(R):
    for j in range(C):
        for k in range(3):
            F[a][b] = img[i][j][k]
            b = b+1
            if k==2:
                F[a][b] = i
                b = b+1
                F[a][b] = j
                b = b+1
        b = 0
        a = a+1

img2 = [[[0 for x in range(3)] for y in range(C)] for z in range(R)]  
img2 = np.asarray(img2) 

while(np.any(F==-1)):
    #randomly choose the mean
    rand_row = randint(0, R*C-1)       
    if F[rand_row][5]==-1:
        for i in range(5):
            seed[i] = F[rand_row][i]
    else:
        continue
    
    F = mean_shift(F, seed)


for i in range(R*C):
    r = F[i][3]
    c = F[i][4]
    for j in range(3):
        img2[r][c][j] = 255-F[i][j]


#show images
plt.subplot(121),plt.imshow(img)
plt.title('Input Image'), plt.xticks([]), plt.yticks([])

plt.subplot(122),plt.imshow(img2)
plt.title('hs=100, hr=100, h=120 '), plt.xticks([]), plt.yticks([])

plt.show()


