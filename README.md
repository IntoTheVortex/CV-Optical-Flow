# Lucas-Kanade Optical Flow
A short Python program implementing the Lucas-Kanade Optical Flow algorithm using NumPy, Matplotlib, and OpenCV-Python (cv2).

--------------------------------------------

This program finds the optical flow from one frame of an image set to the next. First, the two
frames are converted to grayscale, and the difference is taken between them after scaling. The
starting frame is convolved with the two gradient filters, gx and gy, to create the matrices Ix and
Iy. These matrices are used along with the difference to calculate the optical flow vectors with
OLS. The optical flow vectors are then sent with two copies of the starting frame to the plotting
function. One resulting image contains the Vx,Vy vectors plotted, and the other uses circles to
show the magnitude or L2 distance of Vx,Vy. The vectors are plotted every 4 pixels, and the
magnitudes are checked every 4 pixels, only being plotted when the magnitude is greater than
one.  

The first set of images:
![Frame 1](https://github.com/IntoTheVortex/CVDL-Program-2-Optical-Flow/blob/main/frame1_a.png?raw=true)
![Frame 2](https://github.com/IntoTheVortex/CVDL-Program-2-Optical-Flow/blob/main/frame1_b.png?raw=true)  

With optical flow computed and projected on the first frame:
![Result 1](https://github.com/IntoTheVortex/CVDL-Program-2-Optical-Flow/blob/main/first_set.png?raw=true) 

The second set of images:
![Frame 1](https://github.com/IntoTheVortex/CVDL-Program-2-Optical-Flow/blob/main/frame2_a.png?raw=true)
![Frame 2](https://github.com/IntoTheVortex/CVDL-Program-2-Optical-Flow/blob/main/frame2_b.png?raw=true)

With optical flow computed and projected on the first frame:
![Result 2](https://github.com/IntoTheVortex/CVDL-Program-2-Optical-Flow/blob/main/second_set.png?raw=true)

