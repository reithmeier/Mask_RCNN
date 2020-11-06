# **********************************************************************************************************************
#
# brief:    normalize a singe depth file
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




def normalize(arr):
    min = np.min(arr)
    print(min.shape)
    diff = arr - min
    print(diff.shape)
    max = np.max(diff)
    print(max.shape)
    res = diff / max * 255
    print(res.shape)
    return res

def main():
    image = skimage.io.imread(args.input)
    skimage.io.imshow(image)
    plt.show()
    norm = normalize(image)
    skimage.io.imshow(norm)
    plt.show()
    image = skimage.io.imsave(fname=ROOT_DIR + "/logs/norm.png", arr=norm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Path of the input",
                        default="D:\\Data\\elevator\\20181008_105503\\out\\depth\\20181008_105503_000003.png")

    args = parser.parse_args()
    main()
