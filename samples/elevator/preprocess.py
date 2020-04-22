import argparse
import json
import os
import shutil

import cv2
import numpy as np

"""
def resize_dpt(path, file):
    img = cv2.imread(args.input + "/" + path + "/" + file, cv2.IMREAD_UNCHANGED)
    # resize
    width = img.shape[1]
    height = img.shape[0]
    aspect_ratio = float(width) / float(height)
    new_width = int(args.width * aspect_ratio)
    new_height = args.height
    resized = cv2.resize(img, (new_width, new_height))

    # corp
    cropped = np.zeros(shape=[args.height, args.width], dtype=np.uint16)
    w = args.width if resized.shape[1] > args.width else resized.shape[1]
    h = args.height if resized.shape[0] > args.height else resized.shape[0]
    cropped[:h, :w] = resized[:h, :w]
    cv2.imwrite(args.output + "/" + path + "/" + file, cropped)
"""


def resize_dpt(path, file):
    img = cv2.imread(args.input + "/" + path + "/" + file, cv2.IMREAD_UNCHANGED)
    # resize
    width = img.shape[1]
    height = img.shape[0]
    if width > height:
        aspect_ratio = float(height) / float(width)
        new_width = args.width
        new_height = int(args.height * aspect_ratio)
    else:
        aspect_ratio = float(width) / float(height)
        new_width = int(args.width * aspect_ratio)
        new_height = args.height
    resized = cv2.resize(img, (new_width, new_height))

    # crop / buffer
    cropped = np.zeros(shape=[args.height, args.width], dtype=np.uint16)
    w = args.width if resized.shape[1] > args.width else resized.shape[1]
    h = args.height if resized.shape[0] > args.height else resized.shape[0]
    cropped[:h, :w] = resized[:h, :w]
    cv2.imwrite(args.output + "/" + path + "/" + file, cropped)


def resize_img(path, file):
    img = cv2.imread(args.input + "/" + path + "/" + file, cv2.IMREAD_UNCHANGED)
    # resize
    width = img.shape[1]
    height = img.shape[0]
    if width > height:
        aspect_ratio = float(height) / float(width)
        new_width = args.width
        new_height = int(args.height * aspect_ratio)
    else:
        aspect_ratio = float(width) / float(height)
        new_width = int(args.width * aspect_ratio)
        new_height = args.height
    resized = cv2.resize(img, (new_width, new_height))

    # crop / buffer
    cropped = np.zeros(shape=[args.height, args.width, 3], dtype=np.uint8)
    w = args.width if resized.shape[1] > args.width else resized.shape[1]
    h = args.height if resized.shape[0] > args.height else resized.shape[0]
    cropped[:h, :w] = resized[:h, :w]
    cv2.imwrite(args.output + "/" + path + "/" + file, cropped)


def resize_lbl(path, file):
    with open(args.input + "/" + path + "/" + file) as in_file:
        labels = json.load(in_file)
        for completion in labels["completions"]:
            for result in completion["result"]:
                height = result["original_height"]
                width = result["original_width"]
                result["original_height"] = args.height
                result["original_width"] = args.width
                new_points = []
                for pair in result["value"]["points"]:
                    y = pair[0]
                    x = pair[1]
                    if width > height:
                        new_y = y
                        new_x = x * (float(height) / float(width))
                    else:
                        new_y = y * (float(width) / float(height))
                        new_x = x
                    new_points.append([new_y, new_x])
                result["value"]["points"] = new_points
        with open(args.output + "/" + path + "/" + file, "w") as out_file:
            json.dump(labels, out_file)


def mkdir(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)


def main():
    # num_img = sum(1 for line in open(args.input))

    dpt_dir = "depth"
    # dpt_itc_dir = "depth_intrinsics"
    lbl_dir = "labels"
    rgb_dir = "rgb"
    # rgb_itc_dir = "rgb_intrinsics"
    spt_dir = "split"

    mkdir(args.output)
    mkdir(args.output + "/" + dpt_dir)
    # mkdir(args.output + "/" + dpt_itc_dir)
    mkdir(args.output + "/" + lbl_dir)
    mkdir(args.output + "/" + rgb_dir)
    # mkdir(args.output + "/" + rgb_itc_dir)
    mkdir(args.output + "/" + spt_dir)

    # dpt images
    print("preprocess depth images...")
    for filename in os.listdir(args.input + "/" + dpt_dir):
        resize_dpt(dpt_dir, filename)

    # rgb images
    print("preprocess rgb images...")
    for filename in os.listdir(args.input + "/" + rgb_dir):
        resize_img(rgb_dir, filename)

    # labels
    print("preprocess labels...")
    for filename in os.listdir(args.input + "/" + lbl_dir):
        resize_lbl(lbl_dir, filename)

    # index
    print("preprocess index files...")
    for filename in os.listdir(args.input + "/" + spt_dir):
        shutil.copy(args.input + "/" + spt_dir + "/" + filename, args.output + "/" + spt_dir + "/" + filename)

    shutil.copy(args.input + "/" + "all.txt", args.output + "/" + "all.txt")

    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str, help="Path of the output directory", default="./preprocessed")
    parser.add_argument("-i", "--input", type=str, help="Path of the input directory", default="./out")
    parser.add_argument("-w", "--width", type=int, help="Width to resize to", default=512)
    parser.add_argument("-j", "--height", type=int, help="Height to resize to", default=512)

    args = parser.parse_args()
    main()
