import argparse
import os
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import glob


# Parameter
default_root = 'C:/Users/mo/home/_eSDC2_/_PRJ04_/'
default_dir  = default_root + '_2_WIP/_1_forge/_1_coding/'
default_sand = default_root + '_2_WIP/_1_forge/_0_sandbox/'
default_cali = default_root + '_1_INPUT/CarND-Advanced-Lane-Lines-master/camera_cal/'
default_exp  = default_root + '_1_INPUT/CarND-Advanced-Lane-Lines-master/examples/'
default_out  = default_root + '_1_INPUT/CarND-Advanced-Lane-Lines-master/output_images/'
default_test = default_root + '_1_INPUT/CarND-Advanced-Lane-Lines-master/test_images/'

# Helper function: command-line / parse parameters 
def PARSE_ARGS():
    # a b d f g h i j k l m n q r u v w x y z
    parser = argparse.ArgumentParser(prog='advanced techniques for lane finding', description='advanced techniques for lane finding')
    parser.add_argument('-p', '--path',	  dest='path', help='root directory path', action='store', type=str, default=default_dir)
    parser.add_argument('-s', '--sand',	  dest='sand', help='sandbox directory path', action='store', type=str, default=default_sand)
    parser.add_argument('-c', '--cali',	  dest='cali', help='calibration images directory path', action='store', type=str, default=default_cali)
    parser.add_argument('-e', '--exp',	  dest='cali', help='calibration images directory path', action='store', type=str, default=default_exp)
    parser.add_argument('-o', '--out',	  dest='cali', help='calibration images directory path', action='store', type=str, default=default_out)
    parser.add_argument('-t', '--test',	  dest='cali', help='calibration images directory path', action='store', type=str, default=default_test)
    parser.add_argument('-i', '--incorner', dest='tab', help='table size', action='store', type=list, default=[9,6])

    args   = parser.parse_args()
    # a b c d e f g h i j k l m n o p q r s t u v w x y z
    return args


# Helper function: remind me the steps
def steps():
    steps = ['#STEPS',
             'Camera calibration',
             'Distortion correction',
             'Color/gradient threshold',
             'Perspective transform',
             'Detect lane lines',
             'Determine the lane curvature']
    count = None
    for step in steps:
        if count is None:
            print('{}'.format(step))
            count = 1
        else:
            print('{}. {}'.format(count, step))
            count += 1


def main():
    # parameters and placeholders
    args  = PARSE_ARGS()

    # # make a list of calibration images
    # fname = args.cali+'GOPR0042.jpg'
    # img = cv2.imread(fname)
    # plt.imshow(img)

    img = mpimg.imread(default_sand+'traffic_sign_stop.jpg')
    plt.imshow(img)
    plt.show()
    steps()

if __name__ == '__main__':
    main()
