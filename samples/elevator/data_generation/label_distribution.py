# **********************************************************************************************************************
#
# brief:    script to print the distribution of polygon labels over all images
#
# author:   Lukas Reithmeier
# date:     31.07.2020
#
# **********************************************************************************************************************


import argparse
import os
import numpy as np
from matplotlib import pyplot as plt
import skimage.io

ROOT_DIR = os.path.abspath("./../../..")

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def main():
    num_labels = 0
    all_labels = np.zeros((512, 512), np.int64)
    for filename in os.listdir(args.input):
        if filename.endswith(".mask.npy"):
            labels = np.load(os.path.join(args.input, filename))
            num_labels += labels.shape[2]
            label_sum = labels.sum(axis=2)
            # print(label_sum.shape)
            all_labels += label_sum
    """
    for filename in os.listdir(args.input2):
        if filename.endswith(".mask.npy"):
            labels = np.load(os.path.join(args.input, filename))
            num_labels += labels.shape[2]
            label_sum = labels.sum(axis=2)
            # print(label_sum.shape)
            all_labels += label_sum
    """
    fig = skimage.io.imshow(all_labels)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.show()
    print(num_labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Path of the input directory",
                        #default="D:\\Data\\sun_rgbd\\crop\\label13\\train\\")
                        default=ROOT_DIR + "/datasets/elevator/preprocessed/labels/")
    #parser.add_argument("-j", "--input2", type=str, help="Path of the input directory",
                        #default="D:\\Data\\sun_rgbd\\crop\\label13\\test\\")
                         #default=ROOT_DIR + "/datasets/elevator/preprocessed/labels/")
    args = parser.parse_args()
    main()
