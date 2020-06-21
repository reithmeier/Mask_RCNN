import argparse
import json
import os
import time

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
        start_time = time.time()
        hha_img = getHHA(camera_matrix, dpt_img, dpt_img)
        end_time = time.time()
        cv2.imwrite(hha_file, hha_img)
        return end_time - start_time


def main():
    idx_filename = os.path.abspath(args.input + args.file)
    hha_dir = os.path.abspath(args.input + "/hha/")

    time_measurements = []

    if not os.path.exists(hha_dir):
        os.mkdir(hha_dir)

    i = 0

    with open(idx_filename) as in_file:
        for line in in_file:
            if i > 100:
                break
            dpt_file = args.input + line.split(" ")[1][2:].strip()
            itc_file = args.input + line.split(" ")[3][2:].strip()
            hha_file = line.split(" ")[1][8:].strip()
            print(hha_file)
            runtime = process_files(dpt_file, itc_file, hha_dir + hha_file)
            print("elapsed time: ", runtime)
            time_measurements.append(runtime)

            i += 1

    with open("./measurements.txt", 'w') as out_file:
        out_file.write(str(time_measurements))
    with open("./calculations.txt", 'w') as out_file:
        out_file.write("standard deviation: " + str(np.std(time_measurements)))
        out_file.write("mean: " + str(np.mean(time_measurements)))
        out_file.write("median: " + str(np.median(time_measurements)))

    print("runtime measurements: ", time_measurements)
    print("standard deviation: ", np.std(time_measurements))
    print("mean: ", np.mean(time_measurements))
    print("median: ", np.median(time_measurements))
    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, help="index file",
                        default="all.txt")
    parser.add_argument("-i", "--input", type=str, help="Path of the input directory",
                        default=ROOT_DIR + "/datasets/elevator/20181008_105503/out/")
    args = parser.parse_args()
    main()
