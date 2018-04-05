
# coding: utf-8

# # _Prerequisites

# In[1]:

default_dir  = 'C:/Users/mo/home/_eSDC2_/_PRJ04_/_2_WIP/_1_forge/_1_coding/'
default_sand = 'C:/Users/mo/home/_eSDC2_/_PRJ04_/_2_WIP/_1_forge/_0_sandbox/'
default_cali = 'C:/Users/mo/home/_eSDC2_/_PRJ04_/_2_WIP/_1_forge/_0_sandbox/calibration_wide/'


class PARSE_ARGS():
    def __init__(self,
                 path   = default_dir,
                 cali   = default_cali,
                 sand   = default_sand ):

        self.path  = path
        self.cali  = cali
        self.sand  = sand

    def path(self):
        return self.path 
    def cali (self):
        return self.cali
    def sand (self):
        return self.sand
    
args = PARSE_ARGS()


# In[2]:

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
get_ipython().run_line_magic('matplotlib', 'inline')
#%matplotlib qt


# In[3]:

import pandas as pd


# # _L15.9_finding corners

# If you are reading in an image using mpimg.imread():
# - this will read in an RGB image
# - you should convert to grayscale using cv2.COLOR_RGB2GRAY
# 
# But if you are using cv2.imread() or the glob API:
# - this will read in a BGR image
# - you should convert to grayscale using cv2.COLOR_BGR2GRAY

# In[4]:

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')

# prepare object points
nx = 8 # enter the number of inside corners in x
ny = 6 # enter the number of inside corners in y

# Make a list of calibration images
fname = args.cali+'GOPR0042.jpg' # 'test_image.jpg'
img = cv2.imread(fname)    #note: gray = cv2.imread(fname, 0)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Find the chessboard corners
ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

# If found, draw corners
if ret == True:
    # Draw and display the corners
    cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
    plt.imshow(img)


# # _L15.10_calibrating Your Camera
# ### #single image

# In[5]:

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')

# read in a calibration image
img = mpimg.imread(args.cali+'GOPR0040.jpg')
plt.imshow(img)


# In[6]:

# arrays to store object points and image points from all the images
objpoints = [] # 3D points in real world space
imgpoints = [] # 2D points in image plane

# prepare object points, like (0,0,0) , (1,0,0) , (2,0,0) ..., (7,5,0)
objp = np.zeros( (6*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2) # x, y coordinates


# In[7]:

# convert image to grayscale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# find the chessboard corners
ret, corners = cv2.findChessboardCorners(gray, (8,6), None)


# In[ ]:




# In[8]:

# if corners are found, add object points, image points
if ret == True:
    imgpoints.append(corners)
    objpoints.append(objp)
    
    # draw and display the corners
    img = cv2.drawChessboardCorners(img, (8,6), corners, ret )
    plt.imshow(img)    


# In[9]:

type(objpoints)


# ### #several images

# In[10]:

import glob
get_ipython().run_line_magic('matplotlib', 'qt')

# read in a calibration image
images = glob.glob(args.cali+'*.jpg')

# arrays to store object points and image points from all the images
objpoints = [] # 3D points in real world space
imgpoints = [] # 2D points in image plane

# prepare object points, like (0,0,0) , (1,0,0) , (2,0,0) ..., (7,5,0)
objp = np.zeros( (6*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2) # x, y coordinates

for fname in images:
    # read in a calibration image
    img = mpimg.imread(fname)
    
    # convert image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (8,6), None)
    
    # if corners are found, add object points, image points
    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)

        # draw and display the corners
        img = cv2.drawChessboardCorners(img, (8,6), corners, ret )
        plt.imshow(img)  


# ### #notes

# In[ ]:

# opencv to calibrate a camera : cv2.calibrateCamera
# return values: ret, mxt, dist, rvecs, tvecs
# - ret : boolean, true if chesscorners have been found
# - mtx : camera metrics to transform 3D objects to 2D images
# - dist: distorsion coefficient
# - rvecs, tvecs: location of the camera in the world (r: rotation, t: translation)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


# In[ ]:

# return undistorsionned image, also called distination image
# - img: distorted image
# - dst: undistorsionned image, a.k.a distination image
dst = cv2.undistort(img, mtx, dist, None, mtx)


# # _L15.11_Correcting for Distortion

# In[ ]:

import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')

# Read in the saved objpoints and imgpoints
dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )
objpoints = dist_pickle["objpoints"]
imgpoints = dist_pickle["imgpoints"]

# Read in an image
img = cv2.imread(args.cali+'GOPR0040.jpg')

# Write a function that takes an image, object points, and image points
# performs the camera calibration, image distortion correction and 
# returns the undistorted image
def cal_undistort(img, objpoints, imgpoints): # Use cv2.calibrateCamera() and cv2.undistort()
    nx, ny = 5, 5
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # opencv to calibrate a camera :
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # return undistorsionned image, also called distination image
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    # undist = np.copy(img)  # Delete this line
    return undist

undistorted = cal_undistort(img, objpoints, imgpoints)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(undistorted)
ax2.set_title('Undistorted Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


# # _L15.15_Perspective Transform a Stop Sign

# In[11]:

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob


# In[12]:

get_ipython().run_line_magic('matplotlib', 'qt')
# read and display the original image
img = mpimg.imread(args.sand+'traffic_sign_stop.jpg')
plt.imshow(img)


# In[13]:

get_ipython().run_line_magic('matplotlib', 'inline')
# source image points
plt.imshow(img)
plt.plot(51,35, '.')    # top left
plt.plot(45,117, '.')   # bottom left
plt.plot(231,143, '.')  # bottom right
plt.plot(218,81, '.')   # top right


# In[14]:

# define perspective transform function
def warp(img):
    # define calibration box in source (original) and destination (desired and wraped) coordinates
    img_size = (img.shape[1], img.shape[0])
    
    # four source coordinates
    src = np.float32([[51,35],
                      [45,117],
                      [231,143],
                      [218,81]])
    
    # four desired coordinates
    dst = np.float32([[50,35],
                      [50,115],
                      [230,115],
                      [230,35]])
    
    # compute the perspective transform M
    M = cv2.getPerspectiveTransform(src, dst)
    
    # could compute the inverse also by swapping the input parameters
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    # create warped image - uses linear interpolation
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    
    return warped


# In[15]:

nx, ny = 8, 6
#[corners[nx-1][0],corners[nx-1][1]],
#[corners[-1][0],corners[-1][1]],
#[corners[nx*(ny-1)][0],corners[nx*(ny-1)][1]]
[int(corners[nx-1][0]),int(corners[nx-1][1])]


# In[16]:

get_ipython().run_line_magic('matplotlib', 'inline')
# gget perspective transform
warped_im = warp(img)

# visualize undistortion
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

ax1.set_title('source image')
ax1.imshow(img)
ax2.set_title('warped image')
ax2.imshow(warped_im)


# # _L15.17_Undistort and Transform Perspective

# In[ ]:

import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )
#mtx = dist_pickle["mtx"]
#dist = dist_pickle["dist"]
mtx  = np.array([[560.33148363, 0., 651.26264911], [0., 561.3767079, 499.06540191], [0., 0., 1.]])
dist = np.array([[-2.32949182e-01, 6.17242707e-02, - 1.80423444e-05, 3.39635746e-05, -7.54961807e-03]])

# Read in an image
img = cv2.imread('test_image2.png')
nx = 8 # the number of inside corners in x
ny = 6 # the number of inside corners in y

# MODIFY THIS FUNCTION TO GENERATE OUTPUT 
# THAT LOOKS LIKE THE IMAGE ABOVE
def corners_unwarp(img, nx, ny, mtx, dist):
    # Pass in your image into this function
    # Write code to do the following steps
    # 1) Undistort using mtx and dist
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    # 2) Convert to grayscale
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    # 3) Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    # 4) If corners found:
    if ret == True:
        # a) draw corners
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        # b) define 4 source points src = np.float32([[,],[,],[,],[,]])
        src = np.float32([ [corners[0][0][0],corners[0][0][1]],
                           [corners[nx-1][0][0],corners[nx-1][0][1]],
                           [corners[-1][0][0],corners[-1][0][1]],
                           [corners[nx*(ny-1)][0][0],corners[nx*(ny-1)][0][1]]
                        ])
                 #Note: you could pick any four of the detected corners 
                 # as long as those four corners define a rectangle
                 #One especially smart way to do this would be to use four well-chosen
                 # corners that were automatically detected during the undistortion steps
                 #We recommend using the automatic detection of corners in your code
        # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
        dst = np.float32([ [390,300],[890,300],[890,640],[390,640]])
        # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # e) use cv2.warpPerspective() to warp your image to a top-down view
        img_size = (img.shape[1], img.shape[0])
        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    #delete the next two lines
    #M = None
    #warped = np.copy(img) 
    return warped, M

top_down, perspective_M = corners_unwarp(img, nx, ny, mtx, dist)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(top_down)
ax2.set_title('Undistorted and Warped Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


# # _L15.18_How I did it

# In[ ]:

import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Read in an image
img = cv2.imread('test_image2.png')
nx = 8 # the number of inside corners in x
ny = 6 # the number of inside corners in y

# MODIFY THIS FUNCTION TO GENERATE OUTPUT 
# THAT LOOKS LIKE THE IMAGE ABOVE
def corners_unwarp(img, nx, ny, mtx, dist):
    # Pass in your image into this function
    # Write code to do the following steps
    # 1) Undistort using mtx and dist
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    # 2) Convert to grayscale
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    # 3) Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    # 4) If corners found:
    if ret == True:
        # a) draw corners
        cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)
        # b) define 4 source points src = np.float32([[,],[,],[,],[,]])
        src = np.float32([ corners[0],corners[nx-1],corners[-1],corners[nx*(ny-1)] ])   # corners[-nx]]) # corners[nx*(ny-1)] ])
                 #Note: you could pick any four of the detected corners 
                 # as long as those four corners define a rectangle
                 #One especially smart way to do this would be to use four well-chosen
                 # corners that were automatically detected during the undistortion steps
                 #We recommend using the automatic detection of corners in your code
        # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
        img_size = (gray.shape[1], gray.shape[0])
        dist = np.float32([ [100, 100], [img_size[0]-100, 100], [img_size[0]-100, img_size[1]-100], [100, img_size[1]-100] ])
        # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
        M = cv2.getPerspectiveTransform(src, dist)
        # e) use cv2.warpPerspective() to warp your image to a top-down view
        warped = cv2.warpPerspective(undist, M, img_size, flags=cv2.INTER_LINEAR)
    #delete the next two lines
    #M = None
    #warped = np.copy(img) 
    return warped, M

top_down, perspective_M = corners_unwarp(img, nx, ny, mtx, dist)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(top_down)
ax2.set_title('Undistorted and Warped Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


# # _L15.20_Sobel operator

# You need to pass a single color channel to the cv2.Sobel() function,
# so first convert it to grayscale:

# In[17]:

img  = cv2.imread(args.sand+'curved-lane.jpg') # , 0) le 0 pour du gray est KO
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
plt.imshow(gray, cmap='gray')


# **Note:** Make sure you use the correct grayscale conversion depending on how you've read in your images.   
# Use cv2.COLOR_RGB2GRAY if you've read in an image using mpimg.imread().   
# Use cv2.COLOR_BGR2GRAY if you've read in an image using cv2.imread().   

# In[18]:

# Calculate the derivative in the xx direction (the 1, 0 at the end denotes xx direction):
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)

# Calculate the absolute value of the xx derivative:
abs_sobelx = np.absolute(sobelx)

# Convert the absolute value image to 8-bit:
scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))


# Note: It's not entirely necessary to convert to 8-bit (range from 0 to 255) but in practice, it can be useful in the event that you've written a function to apply a particular threshold, and you want it to work the same on input images of different scales, like jpg vs. png. You could just as well choose a different standard range of values, like 0 to 1 etc.

# In[19]:

# Create a binary threshold to select pixels based on gradient strength:
thresh_min = 20
thresh_max = 100
sxbinary = np.zeros_like(scaled_sobel)
sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
plt.imshow(sxbinary, cmap='gray')

# note: pixels have a value of 1 or 0 based on the strength of the x gradient


# In[20]:

# Calculate the derivative in the yy direction (the 0, 1 at the end denotes yy direction):
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

# Calculate the absolute value of the yy derivative:
abs_sobely = np.absolute(sobely)

# Convert the absolute value image to 8-bit:
scaled_sobel = np.uint8(255*abs_sobely/np.max(abs_sobely))

# Create a binary threshold to select pixels based on gradient strength:
thresh_min = 20
thresh_max = 100
sxbinary = np.zeros_like(scaled_sobel)
sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
plt.imshow(sxbinary, cmap='gray')

# note: pixels have a value of 1 or 0 based on the strength of the y gradient


# # _L15.21_Applying Sobel

# In[ ]:

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle


# In[21]:

img  = cv2.imread(args.sand+'thresh-x-example.png') # , 0) le 0 pour du gray est OK :)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # cv2.COLOR_RGB2GRAY)
plt.imshow(gray, cmap='gray')


# #### speudo code

# In[ ]:

# Define a function that applies Sobel x or y, 
# then takes an absolute value and applies a threshold.
# Note: calling your function with orient='x', thresh_min=5, thresh_max=100
# should produce output like the example image shown above this quiz.
def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    # 3) Take the absolute value of the derivative or gradient
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    # 6) Return this mask as your binary_output image
    binary_output = np.copy(img) # Remove this line
    return binary_output


# #### solution

# In[22]:

# Read in an image and grayscale it
image = mpimg.imread(args.sand+'thresh-x-example.png')

# Define a function that applies Sobel x or y, 
# then takes an absolute value and applies a threshold.
# Note: calling your function with orient='x', thresh_min=5, thresh_max=100
# should produce output like the example image shown above this quiz.
def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    # 3) Take the absolute value of the derivative or gradient    
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude 
    # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    # 6) Return this mask as your binary_output image
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return binary_output
    
# Run the function
grad_binary = abs_sobel_thresh(image, orient='x', thresh_min=20, thresh_max=100)


# In[23]:

# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(grad_binary, cmap='gray')
ax2.set_title('Thresholded Gradient', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


# # _L15.22_Magnitude of the Gradient
# #### speudo code

# In[ ]:

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle


# In[24]:

# Read in an image
#image = mpimg.imread('signs_vehicles_xygrad.png')
image = mpimg.imread(args.sand+'thresh-x-example.png')


# In[ ]:

# Define a function that applies Sobel x and y, 
# then computes the magnitude of the gradient
# and applies a threshold
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    # 3) Calculate the magnitude 
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    # 5) Create a binary mask where mag thresholds are met
    # 6) Return this mask as your binary_output image
    binary_output = np.copy(img) # Remove this line
    return binary_output


# #### by MO

# In[25]:

# Define a function that applies Sobel x and y, 
# then computes the magnitude of the gradient
# and applies a threshold
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*magnitude/np.max(magnitude))
    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled_sobel)
    # 6) Return this mask as your binary_output image
    binary_output[(scaled_sobel >= min(mag_thresh)) & (scaled_sobel <= max(mag_thresh))] = 1
    #binary_output = np.copy(img) # Remove this line
    return binary_output


# In[26]:

# Run the function
mag_binary = mag_thresh(image, sobel_kernel=3, mag_thresh=(30, 100))
# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(mag_binary, cmap='gray')
ax2.set_title('Thresholded Magnitude', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


# #### solution

# In[27]:

# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output


# In[ ]:




# # _L15.23_direction of the gradient
# #### speudo code

# In[ ]:

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle


# Read in an image
image = mpimg.imread('signs_vehicles_xygrad.png')

# Define a function that applies Sobel x and y, 
# then computes the direction of the gradient
# and applies a threshold.
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    # 3) Take the absolute value of the x and y gradients
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    # 5) Create a binary mask where direction thresholds are met
    # 6) Return this mask as your binary_output image
    binary_output = np.copy(img) # Remove this line
    return binary_output
    
# Run the function
dir_binary = dir_threshold(image, sobel_kernel=15, thresh=(0.7, 1.3))
# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(dir_binary, cmap='gray')
ax2.set_title('Thresholded Grad. Dir.', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


# #### by MO

# In[ ]:

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle


# In[28]:

# Read in an image
image = mpimg.imread(args.sand+'thresh-x-example.png')


# In[29]:

# Define a function that applies Sobel x and y, 
# then computes the direction of the gradient
# and applies a threshold.
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    grad_orient = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(grad_orient)
    # 6) Return this mask as your binary_output image
    #binary_output = np.copy(img) # Remove this line
    binary_output[(grad_orient >= min(thresh)) & (grad_orient <= max(thresh))] = 1
    return binary_output


# In[30]:

# Run the function
dir_binary = dir_threshold(image, sobel_kernel=15, thresh=(0.7, 1.3))
# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(dir_binary, cmap='gray')
ax2.set_title('Thresholded Grad. Dir.', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


# #### solution

# In[31]:

# Define a function to threshold an image for a given range and Sobel kernel
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output


# # _L15.24_combining Thresholds

# #### speudo code

# In[ ]:

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    # Apply threshold
    return grad_binary

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    # Apply threshold
    return mag_binary

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    # Apply threshold
    return dir_binary


# In[ ]:

# Choose a Sobel kernel size
ksize = 3 # Choose a larger odd number to smooth gradient measurements

# Apply each of the thresholding functions
gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(0, 255))
grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(0, 255))
mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(0, 255))
dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0, np.pi/2))

# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(mag_binary, cmap='gray')
ax2.set_title('Thresholded Magnitude', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


# #### by MO

# In[ ]:

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle


# In[32]:

# Read in an image
image = mpimg.imread(args.sand+'signs_vehicles_xygrad.jpg')
img_solution = mpimg.imread(args.sand+'binary-combo-example.jpg')

# Choose a Sobel kernel size
ksize = 3 # Choose a larger odd number to smooth gradient measurements


# In[61]:

def abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    # 3) Take the absolute value of the derivative or gradient    
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    #return scaled_sobel
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    # 6) Return this mask as your binary_output image
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return binary_output


# In[34]:

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    #return gradmag
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output


# In[35]:

# Define a function to threshold an image for a given range and Sobel kernel
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output


# In[36]:

# Choose a Sobel kernel size
ksize = 9 # Choose a larger odd number to smooth gradient measurements


# In[37]:

# Apply each of the thresholding functions
gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))  # (0, 255))  # (20, 100))
grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(20, 100))
#dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0, np.pi/2))
dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.43, 0.96))


# In[38]:

combined = np.zeros_like(dir_binary)
combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1


# In[39]:

combined2 = np.zeros_like(dir_binary)
combined2[((gradx == 1) & (grady == 1))] = 1

combined3 = np.zeros_like(dir_binary)
#combined3[((mag_binary == 1) & (dir_binary == 1))] = 1
combined3[((gradx == 1) & (grady == 1) & (mag_binary == 1) )] = 1


# Plot the result
figure, axes = plt.subplots(2, 2, figsize=(15, 10))
figure.tight_layout()

axes[0,0].imshow(combined, cmap='gray')
axes[0,0].set_title('combined', fontsize=15)

axes[0,1].imshow(combined2, cmap='gray')
axes[0,1].set_title('combined2', fontsize=15)

axes[1,0].imshow(combined3, cmap='gray')
axes[1,0].set_title('combined3', fontsize=15)

axes[1,1].imshow(dir_binary, cmap='gray')
axes[1,1].set_title('dir_binary', fontsize=15)


# In[ ]:




# In[40]:

# Plot the result
figure, axes = plt.subplots(3, 2, figsize=(15, 10))
figure.tight_layout()

axes[0,0].imshow(image, cmap='gray')
axes[0,0].set_title('Original Image', fontsize=15)


axes[0,1].imshow(img_solution, cmap='gray')
axes[0,1].set_title('Expected result', fontsize=15)
#axes[0,1].imshow(dir_binary, cmap='gray')
#axes[0,1].set_title('dir_binary', fontsize=15)


axes[1,0].imshow(gradx, cmap='gray')
axes[1,0].set_title('gradx', fontsize=15)

axes[1,1].imshow(grady, cmap='gray')
axes[1,1].set_title('grady', fontsize=15)

axes[2,0].imshow(mag_binary, cmap='gray')
axes[2,0].set_title('mag_binary', fontsize=15)

axes[2,1].imshow(combined, cmap='gray')
axes[2,1].set_title('MO: Combined thresholds', fontsize=15)


# # Matrix Product 

# In[41]:

sx = np.array([ [-1,0,1], [-2,0,2], [-1,0,1] ])
sx


# In[42]:

sy = np.array([ [-1,-2,-1], [0,0,0], [1,2,1] ])
sy


# In[43]:

np.dot(sx,sy)


# # _L15.28_HLS and Color Thresholds

# In[ ]:

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# In[44]:

# Read in an image
image = mpimg.imread(args.sand+'curved-lane.jpg')
#img_solution = mpimg.imread(args.sand+'binary-combo-example.jpg')


# #### threshold applied on RGB image

# In[45]:

thresh = (180, 255)
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
binary = np.zeros_like(gray)
binary[(gray > thresh[0]) & (gray <= thresh[1])] = 1


# In[46]:

# Plot the result
figure, axes = plt.subplots(1, 2, figsize=(15, 10))
figure.tight_layout()

axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original Image', fontsize=15)

axes[1].imshow(binary, cmap='gray')
axes[1].set_title('Gray binary', fontsize=15)


# #### explore thresholding individual RGB color channels

# In[47]:

R, G, B = image[:,:,0], image[:,:,1], image[:,:,2]

# Plot the result
figure, axes = plt.subplots(1, 3, figsize=(15, 10))
figure.tight_layout()

axes[0].imshow(R, cmap='gray')
axes[0].set_title('R', fontsize=15)

axes[1].imshow(G, cmap='gray')
axes[1].set_title('G', fontsize=15)

axes[2].imshow(B, cmap='gray')
axes[2].set_title('B', fontsize=15)


# In[91]:

# threshold to find lane-line pixels with R channel as input
thresh = (200, 255)
binary = np.zeros_like(R)
binary[(R > thresh[0]) & (R <= thresh[1])] = 1

# Plot the result
figure, axes = plt.subplots(1, 2, figsize=(15, 10))
figure.tight_layout()

axes[0].imshow(R, cmap='gray')
axes[0].set_title('R channel', fontsize=15)

axes[1].imshow(binary, cmap='gray')
axes[1].set_title('R binary', fontsize=15)


# #### explore thresholding individual H, L, and S color channels

# In[49]:

# Read in an image, convert image in HLS color space, split it per channel
image = mpimg.imread(args.sand+'curved-lane.jpg')
hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
H, L, S = hls[:,:,0], hls[:,:,1], hls[:,:,2]

# Plot the result
figure, axes = plt.subplots(1, 3, figsize=(15, 10))
figure.tight_layout()

axes[0].imshow(H, cmap='gray')
axes[0].set_title('H', fontsize=15)

axes[1].imshow(L, cmap='gray')
axes[1].set_title('L', fontsize=15)

axes[2].imshow(S, cmap='gray')
axes[2].set_title('S', fontsize=15)


# In[50]:

# The S channel picks up the lines well, so let's try applying a threshold there:
thresh = (90, 255)
binary = np.zeros_like(S)
binary[(S > thresh[0]) & (S <= thresh[1])] = 1

# Plot the result
figure, axes = plt.subplots(1, 2, figsize=(15, 10))
figure.tight_layout()

axes[0].imshow(S, cmap='gray')
axes[0].set_title('S channel', fontsize=15)

axes[1].imshow(binary, cmap='gray')
axes[1].set_title('S binary', fontsize=15)


# In[51]:

# H channel, the lane lines appear dark
# so we could try a low threshold there and obtain the following result:
thresh = (15, 100)
binary = np.zeros_like(H)
binary[(H > thresh[0]) & (H <= thresh[1])] = 1

# Plot the result
figure, axes = plt.subplots(1, 2, figsize=(15, 10))
figure.tight_layout()

axes[0].imshow(H, cmap='gray')
axes[0].set_title('H channel', fontsize=15)

axes[1].imshow(binary, cmap='gray')
axes[1].set_title('H binary', fontsize=15)


# #### look at the same thresholds applied to each of these four channels for this image

# In[52]:

# Read in an image
image = mpimg.imread(args.sand+'test4.jpg')


# In[53]:

thresh1, thresh2, thresh3, thresh4 = (180, 255), (200, 255), (90, 255), (15, 100), 
gray 	= cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
hls 	= cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
R, H, S = image[:,:,0], hls[:,:,0], hls[:,:,2]

binary1, binary2, binary3, binary4 = np.zeros_like(gray), np.zeros_like(R), np.zeros_like(S), np.zeros_like(H)
binary1[(gray > thresh1[0]) & (gray <= thresh1[1])] = 1 	# Gray binary
binary2[(R > thresh2[0])    & (R <= thresh2[1])] 	= 1 		# R binary
binary3[(S > thresh3[0])    & (S <= thresh3[1])] 	= 1 		# S binary
binary4[(H > thresh4[0])    & (H <= thresh4[1])] 	= 1 		# H binary

# Plot the result
figure, axes = plt.subplots(4, 2, figsize=(10, 15))
figure.tight_layout()

axes[0,0].imshow(gray, cmap='gray')
axes[0,0].set_title('Gray', fontsize=10)

axes[0,1].imshow(binary1, cmap='gray')
axes[0,1].set_title('Gray binary', fontsize=10)

axes[1,0].imshow(R, cmap='gray')
axes[1,0].set_title('R channel', fontsize=10)

axes[1,1].imshow(binary2, cmap='gray')
axes[1,1].set_title('R binary', fontsize=10)

axes[2,0].imshow(S, cmap='gray')
axes[2,0].set_title('S channel', fontsize=10)

axes[2,1].imshow(binary3, cmap='gray')
axes[2,1].set_title('S binary', fontsize=10)

axes[3,0].imshow(H, cmap='gray')
axes[3,0].set_title('H channel', fontsize=10)

axes[3,1].imshow(binary4, cmap='gray')
axes[3,1].set_title('H binary', fontsize=10)


# In[54]:

combined_new = np.zeros_like(gray)
combined_new[(
             (binary1 == 1)
           & (binary2 == 1)
           & (binary3 == 1)
           & (binary4 == 1)        
            )] = 1

plt.imshow(combined_new, cmap='gray')


# # _L15.29_HLS Quiz

# In[55]:

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# Read in an image, you can also try test1.jpg or test4.jpg
image = mpimg.imread('test4.jpg') 

# Define a function that thresholds the S-channel of HLS
# Use exclusive lower bound (>) and inclusive upper (<=)
def hls_select(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    # 2) Apply a threshold to the S channel
    #H, L, S = hls[:,:,0], hls[:,:,1], hls[:,:,2]
    S = hls[:,:,2]
    binary_output = np.zeros_like(S)
    # 3) Return a binary image of threshold result
    binary_output[(S > thresh[0]) & (S <= thresh[1])] = 1 
    return binary_output
    
hls_binary = hls_select(image, thresh=(90, 255))

# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(hls_binary, cmap='gray')
ax2.set_title('Thresholded S', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


# # _L15.30_combine color and gradient thresholds

# ### The final image color_binary is a combination of:
# - binary thresholding the S channel (HLS)
# - and binary thresholding the result of applying the Sobel operator
# - in the x direction on the original image

# In[69]:

# Read in an image, you can also try test1.jpg or test4.jpg
img = mpimg.imread('test4.jpg') 


# #### original code:

# In[74]:

# Convert to HLS color space and separate the S channel
# Note: img is the undistorted image
hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
s_channel = hls[:,:,2]

# Grayscale image
# NOTE: we already saw that standard grayscaling lost color information for the lane lines
# Explore gradients in other colors spaces / color channels to see what might work better
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Sobel x
sobelx       = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
abs_sobelx   = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

# Threshold x gradient
thresh_min = 20
thresh_max = 100
sxbinary   = np.zeros_like(scaled_sobel)
sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

# Threshold color channel
s_thresh_min = 170
s_thresh_max = 255
s_binary = np.zeros_like(s_channel)
s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1


# In[70]:

# Stack each channel to view their individual contributions in green and blue respectively
# This returns a stack of the two binary images, whose components you can see as different colors
color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

# Combine the two binary thresholds
combined_binary = np.zeros_like(sxbinary)
combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

# Plotting thresholded images
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.set_title('Stacked thresholds')
ax1.imshow(color_binary)

ax2.set_title('Combined S channel and gradient thresholds')
ax2.imshow(combined_binary, cmap='gray')


# #### [MO] OK - replacing piece of code by functions:

# In[75]:

def combine_threshold(sxbinary, s_binary):
    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    # Plotting thresholded images
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.set_title('Stacked thresholds')
    ax1.imshow(color_binary)

    ax2.set_title('Combined S channel and gradient thresholds')
    ax2.imshow(combined_binary, cmap='gray')


# In[76]:

# Convert to HLS color space and separate the S channel
s_binary = hls_select(img, thresh=(170, 255))
# Calculate directional gradient
ksize    = 3
sxbinary = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(20, 100))
# Stack each channel
combine_threshold(sxbinary, s_binary)


# #### Exo: KO : wip : 30. Color and gradient

# In[ ]:

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# In[80]:

image = mpimg.imread(args.sand+'bridge_shadow.png')


# In[83]:

# Edit this function to create your own pipeline.
def pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    return color_binary


# In[84]:

result = pipeline(image)

# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()

ax1.imshow(image)
ax1.set_title('Original Image', fontsize=40)

ax2.imshow(result)
ax2.set_title('Pipeline Result', fontsize=40)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


# # _L15.33_finding the Lines

# #### Line Finding Method: Peaks in a Histogram

# In[92]:

img = np.copy(binary)


# In[ ]:

import numpy as np
histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
plt.figure(figsize=(15,6))
plt.plot(histogram)


# In[111]:

img1 = mpimg.imread(args.sand+'_L15.33_histogram along all the columns by Mo.png')
img2 = mpimg.imread(args.sand+'_L15.33_histogram along all the columns.png')

# Plot the result
f, (ax1, ax2) = plt.subplots(2,1, figsize=(24, 9))
f.tight_layout()

ax1.imshow(img1)
ax1.set_title('By Mo', fontsize=15)

ax2.imshow(img2)
ax2.set_title('Expected result', fontsize=15)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


# #### Implement Sliding Windows and Fit a Polynomial

# In[ ]:

import numpy as np
import cv2
import matplotlib.pyplot as plt

# Assuming you have created a warped binary image called "binary_warped"
# Take a histogram of the bottom half of the image
histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
# Create an output image to draw on and  visualize the result
out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
# Find the peak of the left and right halves of the histogram
# These will be the starting point for the left and right lines
midpoint = np.int(histogram.shape[0]//2)
leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:]) + midpoint

# Choose the number of sliding windows
nwindows = 9
# Set height of windows
window_height = np.int(binary_warped.shape[0]//nwindows)
# Identify the x and y positions of all nonzero pixels in the image
nonzero = binary_warped.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])
# Current positions to be updated for each window
leftx_current = leftx_base
rightx_current = rightx_base
# Set the width of the windows +/- margin
margin = 100
# Set minimum number of pixels found to recenter window
minpix = 50
# Create empty lists to receive left and right lane pixel indices
left_lane_inds = []
right_lane_inds = []

# Step through the windows one by one
for window in range(nwindows):
    # Identify window boundaries in x and y (and right and left)
    win_y_low = binary_warped.shape[0] - (window+1)*window_height
    win_y_high = binary_warped.shape[0] - window*window_height
    win_xleft_low = leftx_current - margin
    win_xleft_high = leftx_current + margin
    win_xright_low = rightx_current - margin
    win_xright_high = rightx_current + margin
    # Draw the windows on the visualization image
    cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
    (0,255,0), 2) 
    cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
    (0,255,0), 2) 
    # Identify the nonzero pixels in x and y within the window
    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
    (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
    (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
    # Append these indices to the lists
    left_lane_inds.append(good_left_inds)
    right_lane_inds.append(good_right_inds)
    # If you found > minpix pixels, recenter next window on their mean position
    if len(good_left_inds) > minpix:
        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
    if len(good_right_inds) > minpix:        
        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

# Concatenate the arrays of indices
left_lane_inds = np.concatenate(left_lane_inds)
right_lane_inds = np.concatenate(right_lane_inds)

# Extract left and right line pixel positions
leftx = nonzerox[left_lane_inds]
lefty = nonzeroy[left_lane_inds] 
rightx = nonzerox[right_lane_inds]
righty = nonzeroy[right_lane_inds] 

# Fit a second order polynomial to each
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)


# #### Visualization

# In[ ]:

# Generate x and y values for plotting
ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
plt.imshow(out_img)
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.xlim(0, 1280)
plt.ylim(720, 0)


# In[102]:

img = mpimg.imread(args.sand+'_L15.33_generate x and y values for plotting.png')
plt.imshow(img)


# #### Skip the sliding windows step once you know where the lines are

# In[ ]:

# Assume you now have a new warped binary image 
# from the next frame of video (also called "binary_warped")
# It's now much easier to find line pixels!
nonzero = binary_warped.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])
margin = 100
left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
left_fit[1]*nonzeroy + left_fit[2] + margin))) 

right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
right_fit[1]*nonzeroy + right_fit[2] + margin)))  

# Again, extract left and right line pixel positions
leftx = nonzerox[left_lane_inds]
lefty = nonzeroy[left_lane_inds] 
rightx = nonzerox[right_lane_inds]
righty = nonzeroy[right_lane_inds]
# Fit a second order polynomial to each
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)
# Generate x and y values for plotting
ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]


# #### And you're done! But let's visualize the result here as well

# In[ ]:

# Create an image to draw on and an image to show the selection window
out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
window_img = np.zeros_like(out_img)
# Color in left and right line pixels
out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

# Generate a polygon to illustrate the search window area
# And recast the x and y points into usable format for cv2.fillPoly()
left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              ploty])))])
left_line_pts = np.hstack((left_line_window1, left_line_window2))
right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
right_line_pts = np.hstack((right_line_window1, right_line_window2))

# Draw the lane onto the warped blank image
cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
plt.imshow(result)
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.xlim(0, 1280)
plt.ylim(720, 0)


# In[101]:

img = mpimg.imread(args.sand+'_L15.33_visualize the result.png')
plt.imshow(img)


# # _L15.34_sliding window search (applying a convolution)

# In[ ]:

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import cv2

# Read in a thresholded image
warped = mpimg.imread('warped_example.jpg')
# window settings
window_width = 50 
window_height = 80 # Break image into 9 vertical layers since image height is 720
margin = 100 # How much to slide left and right for searching

def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def find_window_centroids(image, window_width, window_height, margin):
    
    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions
    
    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 
    
    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(3*image.shape[0]/4):,:int(image.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
    r_sum = np.sum(image[int(3*image.shape[0]/4):,int(image.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(image.shape[1]/2)
    
    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))
    
    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(image.shape[0]/window_height)):
	    # convolve the window into the vertical slice of the image
	    image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height):int(image.shape[0]-level*window_height),:], axis=0)
	    conv_signal = np.convolve(window, image_layer)
	    # Find the best left centroid by using past left center as a reference
	    # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
	    offset = window_width/2
	    l_min_index = int(max(l_center+offset-margin,0))
	    l_max_index = int(min(l_center+offset+margin,image.shape[1]))
	    l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
	    # Find the best right centroid by using past right center as a reference
	    r_min_index = int(max(r_center+offset-margin,0))
	    r_max_index = int(min(r_center+offset+margin,image.shape[1]))
	    r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
	    # Add what we found for that layer
	    window_centroids.append((l_center,r_center))

    return window_centroids

window_centroids = find_window_centroids(warped, window_width, window_height, margin)

# If we found any window centers
if len(window_centroids) > 0:

    # Points used to draw all the left and right windows
    l_points = np.zeros_like(warped)
    r_points = np.zeros_like(warped)

    # Go through each level and draw the windows 	
    for level in range(0,len(window_centroids)):
        # Window_mask is a function to draw window areas
	    l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
	    r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
	    # Add graphic points from window mask here to total pixels found 
	    l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
	    r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

    # Draw the results
    template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
    zero_channel = np.zeros_like(template) # create a zero color channel
    template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
    warpage= np.dstack((warped, warped, warped))*255 # making the original road pixels 3 color channels
    output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results
 
# If no window centers found, just display orginal road image 
else:
    output = np.array(cv2.merge((warped,warped,warped)),np.uint8)

# Display the final results
plt.imshow(output)
plt.title('window fitting results')
plt.show()


# In[112]:

img = mpimg.imread(args.sand+'_L15.33_best window center positions using convolutions.png')
plt.imshow(img)


# # _L15.35_Measuring Curvature

# #### Note:
# - f(y) = Ay^2 + By + C
# - R = [ 1 + (2Ay + B)^2 ]^(3/2) / [ 2A ]
# - yvalue = image.shape[0]   <-- y value corresponding to the bottom of your image   
# 
# #### Note:   
# once the parabola coefficients are obtained, in pixels, convert them into meters.   
# For example, if the parabola is: x= a*(y**2) +b*y+c.     
# And mx and my are the scale for the x and y axis, respectively (in meters/pixel).   
# Then the scaled parabola is: x= mx / (my ** 2) *a*(y**2)+(mx/my)*b*y+c   

# In[115]:

import numpy as np
import matplotlib.pyplot as plt
# Generate some fake data to represent lane-line pixels
ploty = np.linspace(0, 719, num=720)# to cover same y-range as image
quadratic_coeff = 3e-4 # arbitrary quadratic coefficient
# For each y position generate random x position within +/-50 pix
# of the line base position in each case (x=200 for left, and x=900 for right)
leftx = np.array([200 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51) for y in ploty])
rightx= np.array([900 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51) for y in ploty])

leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
rightx= rightx[::-1]  # Reverse to match top-to-bottom in y


# Fit a second order polynomial to pixel positions in each fake lane line
left_fit = np.polyfit(ploty, leftx, 2)
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fit = np.polyfit(ploty, rightx, 2)
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

# Plot up the fake data
mark_size = 3
plt.plot(leftx, ploty, 'o', color='red', markersize=mark_size)
plt.plot(rightx, ploty, 'o', color='blue', markersize=mark_size)
plt.xlim(0, 1280)
plt.ylim(0, 720)
plt.plot(left_fitx, ploty, color='green', linewidth=3)
plt.plot(right_fitx, ploty, color='green', linewidth=3)
plt.gca().invert_yaxis() # to visualize as we do the images


# In[116]:

# Now we have polynomial fits and we can calculate the radius of curvature as follows:

# Define y-value where we want radius of curvature
# I'll choose the maximum y-value, corresponding to the bottom of the image
y_eval = np.max(ploty)
left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
print(left_curverad, right_curverad)
# Example values: 1926.74 1908.48


# #### Note:   
# But now we need to stop and think... We've calculated the radius of curvature based on pixel values, so the radius we are reporting is in pixel space, which is not the same as real world space. So we actually need to repeat this calculation after converting our x and y values to real world space.   
# This involves measuring how long and wide the section of lane is that we're projecting in our warped image. We could do this in detail by measuring out the physical lane in the field of view of the camera, but for this project, you can assume that if you're projecting a section of lane similar to the images above, the lane is about 30 meters long and 3.7 meters wide. Or, if you prefer to derive a conversion from pixel space to world space in your own images, compare your images with U.S. regulations that require a minimum lane width of 12 feet or 3.7 meters, and the dashed lane lines are 10 feet or 3 meters long each.

# In[117]:

# So here's a way to repeat the calculation of radius of curvature 
# after correcting for scale in x and y:

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

# Fit new polynomials to x,y in world space
left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
# Calculate the new radii of curvature
left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
# Now our radius of curvature is in meters
print(left_curverad, 'm', right_curverad, 'm')
# Example values: 632.1 m    626.2 m

# note: http://onlinemanuals.txdot.gov/txdotmanuals/rdw/horizontal_alignment.htm#BGBHGEGC


# # _PRJ04_Drawing

# #### project those lines onto the original image

# In[ ]:

# Create an image to draw the lines on
warp_zero = np.zeros_like(warped).astype(np.uint8)
color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

# Recast the x and y points into usable format for cv2.fillPoly()
pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
pts = np.hstack((pts_left, pts_right))

# Draw the lane onto the warped blank image
cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

# Warp the blank back to original image space using inverse perspective matrix (Minv)
newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
# Combine the result with the original image
result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
plt.imshow(result)


# In[118]:

img = mpimg.imread(args.sand+'lane-drawn.jpg')
plt.imshow(img)


# In[119]:

img = mpimg.imread(args.sand+'_PRJ04_Onward to the Project.PNG')
plt.imshow(img)


# In[ ]:




# In[2]:

inside_corners=[8,6]
nx, ny = inside_corners
print('nx: {}, ny: {}'.format(nx, ny))


# In[ ]:




# # ANNEXE DELETE

# In[ ]:

df_points = pd.DataFrame(points)
df_points.plot.scatter(x=0, y=1)
df_points


# In[ ]:

mtx  = [[560.33148363,0.,651.26264911],[0.,561.3767079,499.06540191],[0.,0.,1.]]
dist = [[-2.32949182e-01, 6.17242707e-02 - 1.80423444e-05, 3.39635746e-05, -7.54961807e-03]]


# In[ ]:

import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read in the saved camera matrix and distortion coefficients
# camera matrix and distortion coefficients
mtx  = np.array([[560.33148363, 0., 651.26264911], [0., 561.3767079, 499.06540191], [0., 0., 1.]])
dist = np.array([[-2.32949182e-01, 6.17242707e-02, - 1.80423444e-05, 3.39635746e-05, -7.54961807e-03]])

# Read in an image
img = cv2.imread('test_image2.png')
nx = 8 # the number of inside corners in x
ny = 6 # the number of inside corners in y

# MODIFY THIS FUNCTION TO GENERATE OUTPUT 
# THAT LOOKS LIKE THE IMAGE ABOVE
def corners_unwarp(img, nx, ny, mtx, dist):
    # 1) Undistort using mtx and dist
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    # 2) Convert to grayscale
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    # 3) Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    # 4) If corners found:
    if ret == True:
        # a) draw corners
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        # b) define 4 source points src = np.float32([[,],[,],[,],[,]])
        src = np.float32([
            [corners[0][0][0],corners[0][0][1]],
            [corners[nx-1][0][0],corners[nx-1][0][1]],
            [corners[-1][0][0],corners[-1][0][1]],
            [corners[nx*(ny-1)][0][0],corners[nx*(ny-1)][0][1]] ])
        # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
        dst = np.float32([[390, 300], [890, 300], [890, 640], [390, 640]])
        # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # e) use cv2.warpPerspective() to warp your image to a top-down view
        img_size = (img.shape[1], img.shape[0])
        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped, M

top_down, perspective_M = corners_unwarp(img, nx, ny, mtx, dist)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(top_down)
ax2.set_title('Undistorted and Warped Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

