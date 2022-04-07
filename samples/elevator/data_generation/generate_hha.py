# **********************************************************************************************************************
#
# brief:    script to generate hha encoded depth images from depth images
#
# author:   Lukas Reithmeier
# date:     04.05.2020
#
# **********************************************************************************************************************


import argparse
import json

import cv2
import numpy as np
from depth2hha.getHHA import *

ROOT_DIR = os.path.abspath("./../../..")


def get_camera_matrix(itc):
    return np.array([[itc["fx"], 0, itc["ppx"]], [0, itc["fy"], itc["ppy"]], [0, 0, 1]])


def process_files(dpt_filename, itc_filename, hha_file):
    with open(itc_filename) as itc_file:
        itc = json.load(itc_file)
        print(itc["ppx"], itc["ppy"], itc["fx"], itc["fy"])
        camera_matrix = get_camera_matrix(itc)
        dpt_img = cv2.imread(dpt_filename, cv2.COLOR_BGR2GRAY) / 10000
        hha_img = getHHA(camera_matrix, dpt_img, dpt_img)
        cv2.imwrite(hha_file, hha_img)


def main():
    idx_filename = os.path.abspath(args.input + args.file)
    hha_dir = os.path.abspath(args.input + "/hha/")

    if not os.path.exists(hha_dir):
        os.mkdir(hha_dir)

    with open(idx_filename + "2", "w") as out_file:
        with open(idx_filename) as in_file:
            for line in in_file:
                dpt_file = args.input + line.split(" ")[1][2:].strip()
                itc_file = args.input + line.split(" ")[3][2:].strip()
                hha_file = line.split(" ")[1][8:].strip()
                print(hha_file)
                process_files(dpt_file, itc_file, hha_dir + hha_file)
                new_line = line.strip() + " ./hha/" + hha_file + "\n"
                print(new_line)
                out_file.write(new_line)

    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, help="index file",
                        default="all.txt")
    parser.add_argument("-i", "--input", type=str, help="Path of the input directory",
                        default=ROOT_DIR + "/datasets/elevator/20181008_105503/out/")
    args = parser.parse_args()
    main()
