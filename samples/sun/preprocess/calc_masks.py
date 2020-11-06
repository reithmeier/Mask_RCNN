# **********************************************************************************************************************
#
# brief:    script to calculate masks from labels and persist them
#
# author:   Lukas Reithmeier
# date:     18.05.2020
#
# **********************************************************************************************************************


import argparse
import os

import cv2
import numpy as np

ROOT_DIR = os.path.abspath("./../../..")


def calculate_mask(lbl_file_name):
    """Generate instance masks for an image.
   Returns:
    masks: A bool array of shape [height, width, instance count] with
        one mask per instance.
    class_ids: a 1D array of class IDs of the instance masks.
    """
    # If not a sun dataset image, delegate to parent class.

    # Convert polygons to a bitmap mask of shape
    # [height, width, instance_count]
    lbl_image = cv2.imread(lbl_file_name, cv2.IMREAD_UNCHANGED)

    height, width = lbl_image.shape[:2]

    mask_found = []
    class_ids = []

    # 13 different classes
    for cls in range(1, 13):
        class_mask = (lbl_image == cls).astype(np.uint8)

        # no instances of cls is present
        if cv2.countNonZero(class_mask) == 0:
            continue

        class_contours, _ = cv2.findContours(class_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for i in range(0, len(class_contours)):
            contour_mask = np.zeros([height, width], dtype=np.uint8)
            cv2.drawContours(contour_mask, class_contours, i, 1, cv2.FILLED)
            mask_found.append(contour_mask)
            class_ids.append(cls)

    # contours found per instance differs from contours found in the label image
    # since the label image consists of numbers of 0-13 in a range of 0-255
    # therefore openCV does not find contours that well
    mask = np.zeros([height, width, len(mask_found)], dtype=np.uint8)
    for i in range(0, len(mask_found)):
        mask[:, :, i] = mask_found[i]

    cv2.waitKey(0)
    return mask, np.array(class_ids)


def process(subdirectory, lbl_file_name):
    mask, class_ids = calculate_mask(lbl_file_name=args.input + subdirectory + lbl_file_name)
    np.save(args.output + subdirectory + lbl_file_name + ".mask.npy", mask)
    np.save(args.output + subdirectory + lbl_file_name + ".class_ids.npy", class_ids)


def main():
    for subdirectory in ["test\\", "train\\"]:
        for filename in os.listdir(args.input + subdirectory):
            if os.path.isfile(args.input + subdirectory + filename) and filename.endswith("png"):
                print(filename)
                process(subdirectory=subdirectory, lbl_file_name=filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str, help="Path of the output directory",
                        default="D:\Data\sun_rgbd\crop\label13\\")
    parser.add_argument("-i", "--input", type=str, help="Path of the input directory",
                        default="D:\Data\sun_rgbd\crop\label13\\")

    args = parser.parse_args()
    main()
