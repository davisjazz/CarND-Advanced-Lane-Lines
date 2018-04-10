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
def parse_args():
    # a b f g h i k l m n q r u v w x y z
    parser = argparse.ArgumentParser(prog='advanced techniques for lane finding', description='advanced techniques for lane finding')
    parser.add_argument('-p', '--path',	  dest='path', help='root directory path', action='store', type=str, default=default_dir)
    parser.add_argument('-s', '--sand',	  dest='sand', help='sandbox directory path', action='store', type=str, default=default_sand)
    parser.add_argument('-c', '--cali',	  dest='cali', help='calibration images directory path', action='store', type=str, default=default_cali)
    parser.add_argument('-e', '--exp',	  dest='exp', help='calibration images directory path', action='store', type=str, default=default_exp)
    parser.add_argument('-o', '--out',	  dest='out', help='calibration images directory path', action='store', type=str, default=default_out)
    parser.add_argument('-t', '--test',	  dest='test', help='calibration images directory path', action='store', type=str, default=default_test)
    parser.add_argument('-i', '--incorner', dest='incorner', help='table size', action='store', type=list, default=[9,6])
    parser.add_argument('-j', '--column', dest='column', help='plot images in a table of n columns', action='store', type=int, default=5)
    parser.add_argument('-d', '--display', dest='to_plot', help='activate the listing of images to plot - by default the listing is not activated', action='store_true')
    # TODO: delete the two following lines in comment
    #parser.add_argument('-t', '--tune', help='activate the fine tune mode - by default it is in training mode', dest='tune', action='store_true', default=False)
    #parser.add_argument('-f', '--freeze', help='freeze all layers except the last one when the fine tune mode is activated - by default these layers are unfrozen', dest='freeze', action='store_false')
    args   = parser.parse_args()
    # a b c d e f g h i j k l m n o p q r s t u v w x y z
    return args


class PARSE_ARGS(object):
    def __init__(self,
                 path = default_dir,
                 sand = default_sand,
                 cali = default_cali,
                 exp  = default_exp,
                 out  = default_out,
                 test = default_test,
                 incorner = [9,6],
                 column   = 5,
                 to_plot  = False):

        self.path = path # root directory path
        self.sand = sand # sandbox directory path
        self.cali = cali # calibration images directory path
        self.exp  = exp  #
        self.out  = out  #
        self.test = test #
        self.incorner = incorner #
        self.column   = column # 
        self.to_plot  = to_plot

    def path(self):
        return self.path
    def sand(self):
        return self.sand
    def cali(self):
        return self.cali
    def exp(self):
        return self.exp
    def out(self):
         return self.out
    def test(self):
        return self.test
    def incorner(self):
        return self.incorner
    def column(self):
        return self.column
    def to_plot(self):
        return self.to_plot

# TODO: delete steps()
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


# TODO: delete template_function()
def template_function():
    '''
    Summary line.

    Extended description of function.

    Parameters
    ----------
    arg1 : int
        Description of arg1
    arg2 : str
        Description of arg2

    Returns
    -------
    int
        Description of return value

    ref: http://docs.python-guide.org/en/latest/writing/documentation/
    '''

    pass


# Helper function: show images
def images_plot(args, images_directory):
    '''
    #parameters:
    . xxx:
        -
    #return:
    . show images
    '''
    # parameters
    image_quantity = len(images_directory)

    if image_quantity%args.column == 0:
        row = image_quantity//args.column
    else:
        row = 1 + (image_quantity//args.column)

    # show images
    figure, axis = plt.subplots(figsize=(15.0, 13.0), nrows=row, ncols=args.column)  # (15.0, 10.0)
    figure.subplots_adjust(hspace=0.2, wspace=0.1)  # hspace=0.2, wspace=0.1) # hspace=0.05, wspace=0.05
    axis = axis.ravel()

    for frame in range(row * args.column):
        if frame < image_quantity:
            # read in a calibration frame
            image = mpimg.imread(images_directory[frame])
            axis[frame].imshow(image)
            axis[frame].axis('off')
            title = images_directory[frame].split('\\')[-1]
            axis[frame].set_title(title, fontsize=8)
        else:
            axis[frame].axis('off')



def main():
    # parameters and placeholders
    args  = parse_args()
    flags = PARSE_ARGS()

    # # make a list of calibration images
    # fname = args.cali+'GOPR0042.jpg'
    # img = cv2.imread(fname)
    # plt.imshow(img)

    # img = mpimg.imread(default_sand+'traffic_sign_stop.jpg')
    # plt.imshow(img)
    # plt.show()
    steps()
    print(args.to_plot)
    print(flags.to_plot)


if __name__ == '__main__':
    main()
