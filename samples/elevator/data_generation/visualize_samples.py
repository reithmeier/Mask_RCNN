# **********************************************************************************************************************
#
# brief:    simple script visualize samples of the elevator dataset
#
# author:   Lukas Reithmeier
# date:     31.07.2020
#
# **********************************************************************************************************************


import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import skimage.io

# Root directory of the project
ROOT_DIR = os.path.abspath("../../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.model import log

from samples.elevator import elevator_rgbd


def main():
    dataset = elevator_rgbd.ElevatorRGBDDataset()
    dataset.load_elevator_rgbd(args.input, "train")
    dataset.prepare()

    # Load random image and mask.
    image_ids = np.random.choice(dataset.image_ids, 10)
    i = 0
    for image_id in image_ids:
        image = dataset.load_image(image_id)
        mask, class_ids = dataset.load_mask(image_id)
        # Compute Bounding box
        bbox = utils.extract_bboxes(mask)

        # Display image and additional stats
        print("image_id ", image_id, dataset.image_reference(image_id))
        log("image", image)
        log("mask", mask)
        log("class_ids", class_ids)
        log("bbox", bbox)
        # Display image and instances
        blank = np.zeros((512, 512, 3), image.dtype)
        print(blank.shape)
        print(image.shape)
        msk_image = visualize.display_instances(blank, bbox, mask, class_ids, dataset.class_names)
        rgb_image = image[:, :, 0:3]
        dpt_image = image[:, :, 3]
        fig = skimage.io.imshow(rgb_image)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.show()
        fig = skimage.io.imshow(dpt_image)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.show()
        print(msk_image.shape)
        print(msk_image.dtype)
        skimage.io.imsave("elevator_dataset_sample_" + str(i) + "_mask.png", msk_image.astype(np.uint8))
        skimage.io.imsave("elevator_dataset_sample_" + str(i) + "_rgb.png", rgb_image)
        skimage.io.imsave("elevator_dataset_sample_" + str(i) + "_dpt.png", dpt_image)

        i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Path of the input directory",
                        default=ROOT_DIR + "/datasets/elevator/preprocessed/")
    args = parser.parse_args()
    main()
