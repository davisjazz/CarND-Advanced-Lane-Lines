import numpy as np
import matplotlib
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


flags = 0

if flags == 1:
    points = np.arange(-5, 5, 0.01)
    dx, dy = np.meshgrid(points, points)
    z = (np.sin(dx)+np.sin(dy))
    plt.imshow(z)
    plt.colorbar()
    plt.title('plot for sin(x)+sin(y)')
    plt.show()
elif flags == 2:
    default_sand = 'C:/Users/mo/home/_eSDC2_/_PRJ04_/_2_WIP/_1_forge/_0_sandbox/'
    img = mpimg.imread(default_sand + '_L15.18_How I did it.png')
    img = cv2.imread(default_sand + '_L15.18_How I did it.png')
    plt.imshow(img)
    plt.show()
elif flags == 3:
    # parameters
    inside_corners = [8, 6]

    # prepare object points
    nx, ny = inside_corners #number of inside corners in x, in y

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


# Helper function: camera calibration
def calibrate(args, inside_corners=[8,6]):
    # prepare object points
    nx, ny = inside_corners #number of inside corners in x, in y

    # read in a calibration image
    images = glob.glob(args.cali + '*.jpg')

    # arrays to store object points and image points from all the images
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

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

            # draw and display the corners
            image = cv2.drawChessboardCorners(image, (nx, ny), corners, ret)
            plt.imshow(image)
            plt.show()

    return objpoints, imgpoints
