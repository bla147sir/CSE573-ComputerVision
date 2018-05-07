import numpy as np
import cv2
from matplotlib import pyplot as plt


#Load an color image in grayscale
img = cv2.imread('/Users/user/Downloads/image.jpg', 0)
#print img

#print type(img) #get the type of img

N = img.shape[0]
M = img.shape[1]
img_shift = [[0 for x in range(M)] for y in range(N)]  
img2 = [[0 for x in range(M)] for y in range(N)]  #temp array to store the result of the first part calculate to reduce the time
img_ft = [[0 for x in range(M)] for y in range(N)] #array has been FT

#shift
for x in range(N):
    for y in range(M):
        img_shift[x][y]=img[x][y]*((-1)**(x+y))

#fourier
for k in range(N):
    for h in range(M):
        for i in range(N):
            img2[k][h]=img2[k][h]+img_shift[i][h]*np.exp(-2*np.pi*1j*(float(k)*i/float(N)))

for k in range(N):
    for h in range(M):
        for j in range(M):
            img_ft[k][h]=img_ft[k][h]+img2[k][j]*np.exp(-2*np.pi*1j*(float(h)*j/float(M)))

img_ft2 = np.log(1+np.abs(img_ft))

#Inverse
img_inverse = img_ft #use the array which hasn't been abs 
img_inverse2 = [[0 for x in range(M)] for y in range(N)] 
img_ift = [[0 for x in range(M)] for y in range(N)] 

for k in range(N):
    for h in range(M):
        for i in range(N):
            img_inverse2[k][h]=img_inverse2[k][h]+1/float(N)*img_inverse[i][h]*np.exp(2*np.pi*1j*(float(k)*i/float(N)))

for k in range(N):
    for h in range(M):
        for j in range(M):
            img_ift[k][h]=img_ift[k][h]+1/float(M)*img_inverse2[k][j]*np.exp(2*np.pi*1j*(float(h)*j/float(M)))

#shift
for x in range(N):
    for y in range(M):
        img_ift[x][y]=img_ift[x][y]*((-1)**(x+y))

img_ift=np.abs(img_ift)

#show images
plt.subplot(131),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(img_ft2, cmap = 'gray')
plt.title('Fourier Transform'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(img_ift, cmap = 'gray')
plt.title('Inverse'), plt.xticks([]), plt.yticks([])
plt.show()

#Mean Square Error
MSE=0
for i in range(N):
    for j in range(M):
        MSE=MSE+(img[i][j]-img_ift[i][j])**2
print 'MSE=', MSE

'''
#verify fourier transform
img_fft = np.fft.fft2(img)

fshift = np.fft.fftshift(img_fft)
#print fshift - img_ft

magnitude_spectrum = 20*np.log(np.abs(fshift))

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude'), plt.xticks([]), plt.yticks([])
plt.show()
#print img_fft
'''