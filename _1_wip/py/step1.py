from step0 import PARSE_ARGS, steps 
import argparse
import os
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import glob




# Helper function: camera calibration
def camera_calibrate(args):
    # prepare object points
    nx, ny = args.incorner #number of inside corners in x, in y

    # read in a calibration image
    images = glob.glob(args.cali + '*.jpg')

    # arrays to store object points and image points from all the images
    objpoints, imgpoints = [], [] # 3D points in real world space, 2D points in image plane

    # prepare object points, like (0,0,0) , (1,0,0) , (2,0,0) ..., (7,5,0)
    objp = np.zeros((nx*ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)  # x, y coordinates

    for frame in images:
        # read in a calibration frame
        image = mpimg.imread(frame)

        # convert frame to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # if corners are found, add object points, frame points
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)

            # # draw and display the corners
            # image = cv2.drawChessboardCorners(image, (nx, ny), corners, ret)
            # plt.imshow(image)
            # plt.show()

    return objpoints, imgpoints

# Helper function:
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

# Helper function: undistort image
def image_undistort(args, image, objpoints, imgpoints):
    # Undistort using mtx and dist
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



def main():
    # parameters and placeholders
    args = PARSE_ARGS()
    objpoints, imgpoints = camera_calibrate(args)

    # return the camera matrix, distortion coefficients, rotation and translation vectors
    corners_found, camera_matrix, coef_distorsion, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    # - corners_found : boolean, true if chesscorners have been found
    # - camera_matrix : camera metrics to transform 3D objects to 2D images
    # - coef_distorsion: distorsion coefficient
    # - rvecs, tvecs: location of the camera in the world (r: rotation, t: translation)


    # image, mtx, dist -> image_undistort

if __name__ == '__main__':
    main()