from module.step0 import PARSE_ARGS, steps
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
    """
    compute the camera matrix and distortion coefficients

    #parameters
    . args.incorner : two dimensional list of int
        - args.incorner[0] is the number of inside corners in x
        - args.incorner[1] is the number of inside corners in y
    . args.cali : str
        - calibration images directory

    #returns
    . camera_matrix : xxx
        - camera metrics to transform 3D objects to 2D images
    . coef_distorsion:
        - distorsion coefficient
    """
    # TODO: delete the following comments
    # . objpoints: list
    #     - 3D points in real world space (args.incorner[0] * args.incorner[1])
    # . imgpoints: list
    #     - 2D points in image plane ( chessboard corners )

    # prepare object points
    nx, ny = args.incorner #number of inside corners in x, in y

    # read in a calibration image
    images = glob.glob(args.cali + '*.jpg')

    # arrays to store object points and image points from all the images
    objpoints, imgpoints = [], [] # 3D points in real world space, 2D points in image plane
    if args.to_plot:
        image_corners_found,  image_corners_not_found = [], []

    # prepare object points, like (0,0,0) , (1,0,0) , (2,0,0) ..., (7,5,0)
    objp = np.zeros((nx*ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)  # x, y coordinates

    for frame in images:
        # read in a calibration frame
        image = mpimg.imread(frame) # RGB for moviepy

        # convert frame to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # if corners are found, add object points, frame points
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)
            if args.to_plot: image_corners_found.append(frame)
        else:
            if args.to_plot: image_corners_not_found.append(frame)


    # TODO: delete the following line
    #return objpoints, imgpoints

    # compute the camera matrix, distortion coefficients, rotation and translation vectors
    # - reprojection_error : the root mean square (RMS) re-projection error
    # - camera_matrix : camera metrics to transform 3D objects to 2D images
    # - coef_distorsion: distorsion coefficient
    # - rvecs, tvecs: location of the camera in the world (r: rotation, t: translation)
    reprojection_error, camera_matrix, coef_distorsion, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    if args.to_plot:
        return [camera_matrix, coef_distorsion], [image_corners_found,  image_corners_not_found]
    else:
        return [camera_matrix, coef_distorsion]

# Helper function:
# TODO: delete or update cal_undistort()
def cal_undistort(img, objpoints, imgpoints): # Use cv2.calibrateCamera() and cv2.undistort()
    '''
    # TODO : document cal_undistort()
    :param img:
    :param objpoints:
    :param imgpoints:
    :return:
    '''
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
def image_undistort(args, image,  camera_matrix, coef_distorsion):
    '''
    # TODO : document image_undistort
    #parameters
    . args:
    . image: array, (x,y,3)
        - image in BRG format, i.e. read the file using cv2.imread()
    . camera_matrix:
    . coef_distorsion:

    #return:
    .
    '''
    # prepare object points
    nx, ny = args.incorner #number of inside corners in x, in y

    # undistort using camera_matrix and coef_distorsion
    undist = cv2.undistort(image, camera_matrix, coef_distorsion, None, camera_matrix)
    # Convert to grayscale
    gray = cv2.cvtColor(undist, cv2.COLOR_RGB2GRAY)
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    # If corners found:
    if ret == True:
        # define 4 source points src = np.float32([[,],[,],[,],[,]])
        src = np.float32([ corners[0],corners[nx-1],corners[-1],corners[nx*(ny-1)] ])   # corners[-nx]]) # corners[nx*(ny-1)] ])
        # define 4 destination points dst = np.float32([[,],[,],[,],[,]])
        image_size = (gray.shape[1], gray.shape[0])
        coef_distorsion = np.float32([ [100, 100], [image_size[0]-100, 100], [image_size[0]-100, image_size[1]-100], [100, image_size[1]-100] ])
        # use cv2.getPerspectiveTransform() to get M, the transform matrix
        M = cv2.getPerspectiveTransform(src, coef_distorsion)
        # use cv2.warpPerspective() to warp your image to a top-down view
        warped = cv2.warpPerspective(undist, M, image_size, flags=cv2.INTER_LINEAR)


    # Return the resulting image and matrix
    return warped, M

def main():
    # parameters and placeholders
    args = PARSE_ARGS()

    # compute the camera matrix, distortion coefficients
    camera_matrix, coef_distorsion = camera_calibrate(args)

    # undistorted image
    image = mpimg.imread(args.cali+'calibration1.jpg')
    image_rectified, matrix_transform = image_undistort(args, image,  camera_matrix, coef_distorsion)

    # show undist image
    plt.imshow(image_rectified)
    plt.show()


if __name__ == '__main__':
    main()