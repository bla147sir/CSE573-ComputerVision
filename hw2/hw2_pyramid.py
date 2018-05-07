import numpy as np
import cv2
from matplotlib import pyplot as plt

#Load an color image in grayscale
img = cv2.imread('/Users/chenshihchia/Desktop/CSE 573 CV/homework/hw2/image3.jpg', 0)

w = [[0 for x in range(5)] for y in range(5)]
wr = [0.05, 0.25, 0.4, 0.25, 0.05]

#5x5 matrix filter
for i in range(5):
    for j in range(5):
        w[i][j] = wr[i]*wr[j]


#convolve the G and kernel and reduce
def g_convo( G ):
    
    G_len = G.shape[0]
    G_width = G.shape[1]
    n = G_len/2
    m = G_width/2
    g = np.zeros((n, m), np.float) #type=array + all elements=0

    for i in range(n):
        for j in range(m):
            for k in range(5):
                for h in range(5):
                    if i*2+k-2<0 or j*2+h-2<0 or i*2+k-2>=G_len or j*2+h-2>=G_width :
                        g[i][j]=g[i][j]
                    else:
                        g[i][j]=g[i][j]+G[i*2+k-2][j*2+h-2]*w[k][h]
    return g
    
G1 = g_convo(img)
G2 = g_convo(G1)
G3 = g_convo(G2)
G4 = g_convo(G3)

#expand the Gaussian
def expand( e ):
    e_len = e.shape[0]
    e_width = e.shape[1]
    n = 2*e_len
    m = 2*e_width
    E = np.zeros((n, m), np.float)
    
    for i in range(n):
        for j in range(m):
            for x in range(5):
                for y in range(5):
                    if (i-x-2)/2<0 or (j-y-2)/2<0 :
                        E[i][j] = E[i][j]
                    else:    
                        E[i][j] = E[i][j]+w[x][y]*e[(i-x-2)/2][(j-y-2)/2]
    return E
    
E1 = expand(G1)
E2 = expand(G2)
E3 = expand(G3)
E4 = expand(G4)

#Laplacian
L1 = img-E1
L2 = G1-E2
L3 = G2-E3
L4 = G3-E4
L5 = G4

#Reconstruct 
R1 = L4+expand(L5)
R2 = L3+expand(R1)
R3 = L2+expand(R2)
R4 = L1+expand(R3)  #original image

#MSE
print 'MSE = ',((img-R4) ** 2).sum()

#show images
plt.subplot(261),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(262),plt.imshow(L1, cmap = 'gray')
plt.title('L1'), plt.xticks([]), plt.yticks([])
plt.subplot(263),plt.imshow(L2, cmap = 'gray')
plt.title('L2'), plt.xticks([]), plt.yticks([])
plt.subplot(264),plt.imshow(L3, cmap = 'gray')
plt.title('L3'), plt.xticks([]), plt.yticks([])
plt.subplot(265),plt.imshow(L4, cmap = 'gray')
plt.title('L4'), plt.xticks([]), plt.yticks([])
plt.subplot(266),plt.imshow(L5, cmap = 'gray')
plt.title('L5(=G4)'), plt.xticks([]), plt.yticks([])

plt.subplot(267),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(268),plt.imshow(R1, cmap = 'gray')
plt.title('R1'), plt.xticks([]), plt.yticks([])
plt.subplot(269),plt.imshow(R2, cmap = 'gray')
plt.title('R2'), plt.xticks([]), plt.yticks([])
plt.subplot(2,6,10),plt.imshow(R3, cmap = 'gray')
plt.title('R3'), plt.xticks([]), plt.yticks([])
plt.subplot(2,6,11),plt.imshow(R4, cmap = 'gray')
plt.title('R4'), plt.xticks([]), plt.yticks([])

plt.show()


