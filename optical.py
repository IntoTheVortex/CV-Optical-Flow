#Author: Amber Shore
#Version: 2022-07-16

import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv


def get_image_padded(image, p):
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
    data = get_image_padded(data, p)
    f = len(_filter)
    new_shape_i = (n + 2 * p - f + 1)
    new_shape_j = (m + 2 * p - f + 1)
    convolved = np.zeros(shape=(new_shape_i,new_shape_j))

    #stride = 1
    for i in range(0,n):
        for j in range(0,m):
            if i+f < n and j+f < m:
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
    f = 3 

    V = np.zeros([n,m,2])
    for i in range(0,n-2):
        for j in range(0,m-2):
            if i+f < n and j+f < m:
                #get A, then compute A transpose times A:
                ax = np.asarray(Ix[i:i+f, j:j+f].flatten()).T 
                ay = np.asarray(Iy[i:i+f, j:j+f].flatten()).T
                A = np.stack((ax,ay), axis=1)
                AtA = np.matmul(A.T, A)

                #get inverse of AtA
                try:
                    inv_AtA = np.linalg.inv(AtA)
                except:
                    inv_AtA = np.linalg.pinv(AtA)

                #get Atb, then complete calculation V = AtA * Atb
                b = np.asarray(Itp[i:i+f, j:j+f]).flatten().T
                Atb = np.matmul(A.T, b)
                V[i][j] = np.matmul(inv_AtA, Atb)

    return V

def plot_vectors(image, l2_image, V, title):
    color = (0,255,0)
    for x in range(0, len(V), 4):
        for y in range(0, V.shape[1]-2, 4):
            if V[x][y][0] > 0 or V[x][y][1] > 0:
                x1 = int(x + V[x][y][0])
                y1 = int(y + V[x][y][1]) 
                mag = np.sqrt(pow((y1-y),2) + pow((x1-x),2))

                #cv2 read in width by height instead of height by width,
                # so x and y are switched for plotting
                image = cv.line(image, (y,x), (y1,x1), color=color, thickness=1)
                if(mag > 1):
                    #magnitude is divided by half so that it can represent the diameter,
                    # instead of the radius
                    l2_image = cv.circle(l2_image, (y,x), int(mag/2), color=color, thickness=1)

    cv.imwrite(title, image)
    cv.imwrite("l2_"+title, l2_image)

def process_image_pair(name1, name2, title):
    p = 1
    image1 = cv.imread(name1, cv.IMREAD_GRAYSCALE)
    color_image = cv.imread(name1, cv.IMREAD_COLOR)
    l2_color_image = cv.imread(name1, cv.IMREAD_COLOR)
    image2 = cv.imread(name2, cv.IMREAD_GRAYSCALE)

    Itp = get_pixel_intensity_difference(image1, image2) #scaled here
    gx, gy = create_dfilters()
    Ix = convolve(image1, gx, p) 
    Iy = convolve(image1, gy, p) 
    V = ols_flow(Ix, Iy, Itp)
    plot_vectors(color_image, l2_color_image, V, title)


def main():
    im1_name1 = 'frame1_a.png'
    im1_name2 = 'frame1_b.png'

    im2_name1 = 'frame2_a.png'
    im2_name2 = 'frame2_b.png'

    process_image_pair(im1_name1, im1_name2, 'first_set.png')
    process_image_pair(im2_name1, im2_name2, 'second_set.png')


if __name__ == '__main__':
    main()

