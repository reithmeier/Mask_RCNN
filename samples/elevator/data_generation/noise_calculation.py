# **********************************************************************************************************************
#
# brief:    script to calculate a noise image from depth images
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


ROOT_DIR = os.path.abspath("./../../..")


def read_all_images():
    images = []
    i = 0
    for root, dirs, files in os.walk(args.input):
        for file in files:
            i += 1
            if i < 89:
                continue
            if i > 100:
                break
            print(file)
            image = skimage.io.imread(args.input + "/" + file)

            images.append(image)
    return np.array(images)




def main():
    images = read_all_images()
    print(images.shape)
    mean = np.mean(images, axis=0)
    skimage.io.imshow(mean)
    plt.show()
    avg = np.average(images, axis=0)
    skimage.io.imshow(avg)
    plt.show()
    std = np.std(images, axis=0)
    skimage.io.imshow(std)
    plt.show()
    print(std.shape)
    std_flat = std.flatten()
    print(std_flat.shape)
    print(std_flat)
    print(args.output, args.filename)
    np.save(file=args.output + "/" + args.filename, arr=std)

    fig = plt.figure(figsize=(10, 7))
    # Creating plot
    plt.boxplot(std_flat)
    # show plot
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str, help="Path of the output directory",
                        default=ROOT_DIR + "/logs/noise")
    parser.add_argument("-f", "--filename", type=str, help="Path of the output directory",
                        default="20190219_132346.npy")
    parser.add_argument("-i", "--input", type=str, help="Path of the input directory",
                        default="D:\\Data\\elevator\\20190219_132346\\out\\depth")

    args = parser.parse_args()
    main()
