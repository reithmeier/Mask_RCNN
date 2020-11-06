# **********************************************************************************************************************
#
# brief:    script to preprocess the data
#
# author:   Lukas Reithmeier
# date:     09.07.2020
#
# **********************************************************************************************************************

import numpy as np
from skimage import io
from mrcnn import utils

INPUT_DIR = "D:\Data\sun_rgbd\\crop\\"
OUTPUT_DIR = "D:\Data\sun_rgbd\\resized\\"

NUM_TRAIN = 5285
NUM_TEST = 5050

IMAGE_MIN_DIM = 256
IMAGE_MAX_DIM = 256
IMAGE_RESIZE_MODE = "square"
IMAGE_MIN_SCALE = 0


def resize(filename):
    file_path = INPUT_DIR + filename
    image = io.imread(file_path)
    molded_image, window, scale, padding, crop = utils.resize_image(
        image,
        min_dim=IMAGE_MIN_DIM,
        max_dim=IMAGE_MAX_DIM,
        min_scale=IMAGE_MIN_SCALE,
        mode=IMAGE_RESIZE_MODE
    )
    output_path = OUTPUT_DIR + filename
    io.imsave(output_path, molded_image)


def resize_masks(filename):
    file_path = INPUT_DIR + filename
    print(file_path)
    image = np.load(file_path)
    if image.shape[2] > 0:
        molded_image, window, scale, padding, crop = utils.resize_image(
            image,
            min_dim=IMAGE_MIN_DIM,
            max_dim=IMAGE_MAX_DIM,
            min_scale=IMAGE_MIN_SCALE,
            mode=IMAGE_RESIZE_MODE
        )
    else:
        molded_image = image
    output_path = OUTPUT_DIR + filename
    np.save(output_path, molded_image)


def reformat(filename):
    input_path = OUTPUT_DIR + filename
    output_path = OUTPUT_DIR + filename[:-9]
    print(input_path)
    print(output_path)

    img = np.load(input_path)
    io.imsave(output_path, img)


if __name__ == '__main__':
    """
    # train
    for i in range(1, NUM_TRAIN + 1):
        print("0000{:04}.png".format(i))
        resize("depth\\train\\0000{:04}.png".format(i))
        resize("image\\train\\img-00{:04}.jpg".format(i))
        resize_masks("label13\\train\\img13labels-00{:04}.png.mask.npy".format(i))

    # test
    for i in range(1, NUM_TEST + 1):
        print("0000{:04}.png".format(i))
        resize("depth\\test\\0000{:04}.png".format(i))
        resize("image\\test\\img-00{:04}.jpg".format(i))
        resize_masks("label13\\test\\img13labels-00{:04}.png.mask.npy".format(i))
    """
    # reformat
    for i in range(1, NUM_TRAIN + 1):
        reformat("label13\\test\\img13labels-00{:04}.png.mask.npy".format(i))
    for i in range(1, NUM_TEST + 1):
        reformat("label13\\test\\img13labels-00{:04}.png.mask.npy".format(i))
