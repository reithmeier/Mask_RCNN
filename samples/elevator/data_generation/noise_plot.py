# **********************************************************************************************************************
#
# brief:    script to plot boxplots of std noise images
#
# author:   Lukas Reithmeier
# date:     18.05.2020
#
# **********************************************************************************************************************


import argparse
import json
import os

import numpy as np
import skimage.io
from matplotlib import pyplot as plt
import  matplotlib

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


def read_all_images():
    images = []
    for root, dirs, files in os.walk(args.input):
        for file in files:
            # print(file)
            if not file.endswith(".npy"):
                continue
            image = np.load(args.input + "/" + file)
            #fig = skimage.io.imshow(image)
            #fig.axes.get_xaxis().set_visible(False)
            #fig.axes.get_yaxis().set_visible(False)
            #plt.show()
            image = image.flatten()
            images.append(image)
            # Creating plot
            # plt.boxplot(image)
            # show plot
            # plt.show()

    return np.array(images)

def hist(arr, i):
    with plt.style.context('ggplot'):
        fig, ax = plt.subplots()
        ax.set_yscale("log")
        ax.hist(arr, bins=10, edgecolor="w", linewidth=1, color="CornFlowerBlue")
        ax.set_ylabel('Frequency', fontsize=BIGGER_SIZE)
        ax.set_xlabel('Distance [mm]', fontsize=BIGGER_SIZE)
    from tikzplotlib import save as tikz_save
    tikz_save(f"noise_dist_{i}.tex")
    plt.show()

def main():
    images = read_all_images()

    # Creating plot
    with plt.style.context('ggplot'):
        flierprops = dict(marker='.', markerfacecolor='r', markersize=1,
                          linestyle='none', markeredgecolor='g')
        fig, ax = plt.subplots()
        # fig.set_size_inches(16,32)
        ax.set_yscale("log")
        ax.boxplot(images, flierprops=flierprops)
    # show plot
    plt.show()
    # Creating plot
    with plt.style.context('ggplot'):
        fig, ax = plt.subplots()
        ax.set_yscale("log")
        ax.boxplot(images, showfliers=False)
    # show plot
    plt.show()

    hist(images[0], 1)
    hist(images[1], 2)
    hist(images[2], 3)
    hist(images[3], 4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Path of the input directory",
                        default=ROOT_DIR + "/logs/noise")

    args = parser.parse_args()
    main()
