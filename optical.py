#Author: Amber Shore
#Version: 2022-07-08

import random
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv



def get_image_padded(image):
    p = 1 #how many zeros to pad the matrix with
        
    image = image/255 #normalize
    image_padded = np.pad(image, ((p,p),(p,p)), 'constant')

    return image_padded 

def get_pixel_intensity_difference(image1, image2):
    image1 = np.asarray(image1/255, dtype=np.float64)
    image2 = np.asarray(image2/255, dtype=np.float64)
    difference = image2 - image1
    return difference

def convolve(data, _filter, p):
    n = len(data)
    m = len(data[0])
    f = len(_filter)
    new_shape_i = (n + 2 * p - f + 1)
    new_shape_j = (m + 2 * p - f + 1)
    convolved = np.zeros(shape=(new_shape_i,new_shape_j))

    #stride = 1
    for i in range(0,n):
        for j in range(0,m):
            if i+f > n or j+f > m:
                continue
            a = np.asarray(data[i:i+f, j:j+f])
            convolved[i][j] = sum(sum(a * _filter))

    return convolved


def create_dfilters():
    gx = np.asarray([[1, 0, -1], 
                     [2, 0, -2], 
                     [1, 0, -1]])
    gy = np.asarray([[1, 2, 1], 
                     [0, 0, 0], 
                     [-1, -2, -1]])
    return gx, gy

def ols_flow(Ix, Iy, Itp):
    n = len(Ix)
    m = len(Ix[0])
    f = 2 

    print('ix shape:', Ix.shape)
    V = np.zeros([n,m,2])
    print('v shape:', V.shape)
    print('it shape:', Itp.shape)
    for i in range(1,n-2):
        for j in range(1,m-2):
            if i+f > (n-2) or j+f > (m-2):
                continue
            else:
                #get A, then compute A transpose A:
                #print('i:',i, 'j:',j, 'ix:',Ix[i][j], 'iy',Iy[i][j])
                ax = np.asarray(Ix[i-1:i+f, j-1:j+f].flatten()).T 
                ay = np.asarray(Iy[i-1:i+f, j-1:j+f].flatten()).T
                A = np.stack((ax,ay), axis=1)
                #print('A:81')
                #print(A.shape)
                #print(A)
                AtA = np.matmul(A.T, A)
                #print('AtA:86')
                #print(AtA.shape)

                #get inverse of AtA
                try:
                    inv_AtA = np.linalg.inv(AtA)
                except:
                    inv_AtA = np.linalg.pinv(AtA)

                #get Atb, then complete calculation V = AtA * Atb
                b = np.asarray(Itp[i-1:i+f, j-1:j+f]).flatten().T
                Atb = np.matmul(A.T, b)
                V[i][j] = np.matmul(inv_AtA, Atb)

    return V

def plot_vectors(image, V):
    #vectors = np.stack((np.arange()))
    color = (0,50,250)
    for x in range(0, len(V), 4):
        for y in range(0, V.shape[1]-2, 4):
            if V[x][y][0] > 0 or V[x][y][1] > 0:
                x1 = int(x + V[x][y][0])
                y1 = int(y + V[x][y][1]) 
                #print('x', x1, 'y', y1)
                #cv.arrowedLine(image, (x,y), (x1,y1), color=color)
                mask = cv.line(image, (x,y), (x1,y1), color=(0,.8,0), thickness=1)

    image = cv.add(image, mask)
    #image = np.asarray(np.clip(image*255, 0, 255), dtype=np.uint8);
    cv.imwrite("test.png", image)
    

def main():
    name1 = 'frame1_a.png'
    name2 = 'frame1_b.png'
    image1 = cv.imread(name1, cv.IMREAD_GRAYSCALE)
    image2 = cv.imread(name2, cv.IMREAD_GRAYSCALE)
    image_padded = get_image_padded(image1)

    Itp = get_pixel_intensity_difference(image1, image2)
    gx, gy = create_dfilters()
    Ix = convolve(image_padded, gx, 1) 
    Iy = convolve(image_padded, gy, 1) 
    V = ols_flow(Ix, Iy, Itp)
    #print(V)
    plot_vectors(image1, V)


    #cv.imwrite('result.jpg', result)



if __name__ == '__main__':
    main()

