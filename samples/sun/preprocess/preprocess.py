# **********************************************************************************************************************
#
# brief:    script to preprocess the data
#
# author:   Lukas Reithmeier
# date:     19.04.2020
#
# **********************************************************************************************************************

import cv2
import numpy as np

DATA_DIR = "I:\Data\sun_rgbd\\"
OUTPUT_DIR_BUFFER = "I:\Data\sun_rgbd\\buffer\\"
OUTPUT_DIR_CROP = "I:\Data\sun_rgbd\\crop\\"

NUM_TRAIN = 5285
NUM_TEST = 5050


def buffer_dpt(path, file):
    img = cv2.imread(DATA_DIR + path + file, cv2.IMREAD_UNCHANGED)
    new = np.zeros(shape=[1024, 1024], dtype=np.uint16)
    new[:img.shape[0], :img.shape[1]] = img
    cv2.imwrite(OUTPUT_DIR_BUFFER + path + file, new)


def buffer_img(path, file):
    img = cv2.imread(DATA_DIR + path + file, cv2.IMREAD_UNCHANGED)
    new = np.zeros(shape=[1024, 1024, 3], dtype=np.uint16)
    new[:img.shape[0], :img.shape[1]] = img
    cv2.imwrite(OUTPUT_DIR_BUFFER + path + file, new)


def buffer_lbl(path, file):
    img = cv2.imread(DATA_DIR + path + file, cv2.IMREAD_UNCHANGED)
    new = np.zeros(shape=[1024, 1024], dtype=np.uint8)
    new[:img.shape[0], :img.shape[1]] = img
    cv2.imwrite(OUTPUT_DIR_BUFFER + path + file, new)


def crop_dpt(path, file):
    img = cv2.imread(DATA_DIR + path + file, cv2.IMREAD_UNCHANGED)
    new = np.zeros(shape=[512, 512], dtype=np.uint16)
    w = 512 if img.shape[1] > 512 else img.shape[1]
    h = 512 if img.shape[0] > 512 else img.shape[0]
    new[:h, :w] = img[:h, :w]
    cv2.imwrite(OUTPUT_DIR_CROP + path + file, new)


def crop_img(path, file):
    img = cv2.imread(DATA_DIR + path + file, cv2.IMREAD_UNCHANGED)
    new = np.zeros(shape=[512, 512, 3], dtype=np.uint8)
    w = 512 if img.shape[1] > 512 else img.shape[1]
    h = 512 if img.shape[0] > 512 else img.shape[0]
    new[:h, :w] = img[:h, :w]
    cv2.imwrite(OUTPUT_DIR_CROP + path + file, new)


def crop_lbl(path, file):
    img = cv2.imread(DATA_DIR + path + file, cv2.IMREAD_UNCHANGED)
    new = np.zeros(shape=[512, 512], dtype=np.uint8)
    w = 512 if img.shape[1] > 512 else img.shape[1]
    h = 512 if img.shape[0] > 512 else img.shape[0]
    new[:h, :w] = img[:h, :w]
    cv2.imwrite(OUTPUT_DIR_CROP + path + file, new)


if __name__ == '__main__':

    # train
    for i in range(1, NUM_TRAIN + 1):
        print("0000{:04}.png".format(i))
        buffer_dpt("depth\\train\\", "0000{:04}.png".format(i))
        buffer_img("image\\train\\", "img-00{:04}.jpg".format(i))
        buffer_lbl("label13\\train\\", "img13labels-00{:04}.png".format(i))
        crop_dpt("depth\\train\\", "0000{:04}.png".format(i))
        crop_img("image\\train\\", "img-00{:04}.jpg".format(i))
        crop_lbl("label13\\train\\", "img13labels-00{:04}.png".format(i))

    # test
    for i in range(1, NUM_TEST + 1):
        print("0000{:04}.png".format(i))
        buffer_dpt("depth\\test\\", "0000{:04}.png".format(i))
        buffer_img("image\\test\\", "img-00{:04}.jpg".format(i))
        buffer_lbl("label13\\test\\", "img13labels-00{:04}.png".format(i))
        crop_dpt("depth\\test\\", "0000{:04}.png".format(i))
        crop_img("image\\test\\", "img-00{:04}.jpg".format(i))
        crop_lbl("label13\\test\\", "img13labels-00{:04}.png".format(i))
