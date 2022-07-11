#Author: Amber Shore
#Version: 2022-07-08

import random
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv



def get_image_grayscale(image_file):
    p = 1 #how many zeros to pad the matrix with

    image = cv.imread(image_file, cv.IMREAD_GRAYSCALE)

    h, w = image.shape
    '''don't think I'll need to resize
    while (h > 1600 or w> 1600):
        image = cv.resize(image, None, fx=0.9, fy=0.9, interpolation=cv.INTER_LINEAR)
        h, w, c = image.shape
    '''
        
    image = image/255 #normalize
    image_padded = np.pad(image, ((p,p),(p,p)), 'constant')

    return image_padded 

def get_pixel_intensity_difference(image1, image2):
    difference = image2 - image1
    #i thought it would be more involved
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
    for i in range(1,n-1):
        for j in range(1,m-1):
            if i+f > n or j+f > m:
                continue
            else:
                #get A, then compute A transpose A:
                #pixel_group = np.asarray(Ix[i-1:i+f, j-1:j+f]).flatten().T
                #A = pixel_group.T
                print('i:',i, 'j:',j, 'ix:',Ix[i][j], 'iy',Iy[i][j])
                #A = np.stack((pixel_group, pixel_group), axis=1)
                #A = A * Ix[i:(i-1)+f, j:(j-1)+f]
                ax = np.asarray(Ix[i-1:i+f, j-1:j+f].flatten()).T 
                ay = np.asarray(Iy[i-1:i+f, j-1:j+f].flatten()).T
                A = np.stack((ax,ay), axis=1)
                print('A:81')
                print(A.shape)
                print(A)
                #A = np.concatenate(
                    #np.matmul(A[:,0], Ix[i-1:i+f, j-1:j+f]), 
                    #np.matmul(A[:,1], Iy[i-1:i+f, j-1:j+f]))
                AtA = np.matmul(A.T, A)
                #print('AtA:86')
                #print(AtA.shape)
                #get inverse of AtA
                try:
                    inv_AtA = np.linalg.inv(AtA)
                except:
                    inv_AtA = np.linalg.pinv(AtA)

                b = np.asarray(Itp[i-1:i+f, j-1:j+f]).flatten().T
                #b = np.matmul(b, pixel_group)
                Atb = np.matmul(A.T, b)
                V[i][j] = np.matmul(inv_AtA, Atb)

    return V
    

def main():
    name1 = 'frame1_a.png'
    name2 = 'frame1_b.png'
    image1 = get_image_grayscale(name1)
    image2 = get_image_grayscale(name2)

    Itp = get_pixel_intensity_difference(image1, image2)
    gx, gy = create_dfilters()
    Ix = convolve(image1, gx, 1) 
    Iy = convolve(image1, gy, 1) 
    V = ols_flow(Ix, Iy, Itp)
    print(V)


    #cv.imwrite('result.jpg', result)



if __name__ == '__main__':
    main()

