import argparse
import json
import os

import cv2
import numpy as np

ROOT_DIR = os.path.abspath("./../../..")


def resize_dpt(file):
    img = cv2.imread(args.input + file[1:], cv2.IMREAD_UNCHANGED)
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

    # normalize
    normalized = np.zeros(shape=[args.height, args.width], dtype=np.uint16)
    cv2.normalize(cropped, normalized, np.iinfo(np.uint16).max, 0, cv2.NORM_MINMAX)
    cv2.imwrite(args.output + file[1:], normalized)


def resize_img(file):
    img = cv2.imread(args.input + file[1:], cv2.IMREAD_UNCHANGED)
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
    cv2.imwrite(args.output + file[1:], cropped)


def resize_lbl(file):
    with open(args.input + file[1:]) as in_file:
        labels = json.load(in_file)
        for completion in labels["completions"]:
            for result in completion["result"]:
                if result["type"] != "polygonlabels":
                    continue
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
        with open(args.output + file[1:], "w") as out_file:
            json.dump(labels, out_file)


def contains_labels(lbl_filename):
    with open(lbl_filename) as lbl_file:
        data = json.load(lbl_file)
        annotations = data['completions'][-1]["result"]
        return len(annotations) > 0


def requirements_met(line):
    rgb_filename = line.split(" ")[0].strip()
    dpt_filename = line.split(" ")[1].strip()
    rgb_itc_filename = line.split(" ")[2].strip()
    dpt_itc_filename = line.split(" ")[3].strip()
    lbl_filename = line.split(" ")[4].strip()
    return os.path.exists(args.input + rgb_filename) and os.path.isfile(args.input + rgb_filename) and \
           os.path.exists(args.input + dpt_filename) and os.path.isfile(args.input + dpt_filename) and \
           os.path.exists(args.input + rgb_itc_filename) and os.path.isfile(args.input + rgb_itc_filename) and \
           os.path.exists(args.input + dpt_itc_filename) and os.path.isfile(args.input + dpt_itc_filename) and \
           os.path.exists(args.input + lbl_filename) and os.path.isfile(args.input + lbl_filename) and \
           contains_labels(args.input + lbl_filename)


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
    i = 0
    with open(args.input + "/" + "all.txt") as in_file:
        with open(args.output + "/" + "all.txt", "w") as out_file:
            for line in in_file:
                if requirements_met(line):
                    print(i)
                    out_file.write(line)
                    rgb_filename = line.split(" ")[0].strip()
                    dpt_filename = line.split(" ")[1].strip()
                    rgb_itc_filename = line.split(" ")[2].strip()
                    dpt_itc_filename = line.split(" ")[3].strip()
                    lbl_filename = line.split(" ")[4].strip()
                    resize_img(rgb_filename)
                    resize_dpt(dpt_filename)
                    resize_lbl(lbl_filename)
                else:
                    print("skip", i)
                i = i + 1
    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str, help="Path of the output directory",
                        default=ROOT_DIR + "/datasets/elevator/N_18-3_2-0_2/preprocessed")
    parser.add_argument("-i", "--input", type=str, help="Path of the input directory",
                        default=ROOT_DIR + "/datasets/elevator/N_18-3_2-0_2/out")
    parser.add_argument("-w", "--width", type=int, help="Width to resize to", default=512)
    parser.add_argument("-j", "--height", type=int, help="Height to resize to", default=512)

    args = parser.parse_args()
    main()
