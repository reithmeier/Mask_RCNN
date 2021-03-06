# **********************************************************************************************************************
#
# brief:    Mask R-CNN
#           Configurations and data loading code for the sun rgbd dataset.
#           uses rgbd data
#           see: https://github.com/matterport/Mask_RCNN/wiki#training-with-rgb-d-or-grayscale-images
#
# author:   Lukas Reithmeier
# date:     19.04.2020
#
# **********************************************************************************************************************

import os
import sys

import numpy as np
import skimage.draw

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn import utils

from samples.sun.sunrgb import SunRGBConfig


class SunRGBDConfig(SunRGBConfig):
    """Configuration for training on the sun rgbd dataset.
    Derives from the base Config class and overrides values specific
    to the sun rgbd dataset.
    """
    # Give the configuration a recognizable name
    NAME = "sunrgbd"

    # 3 color channels +  1 depth channel
    IMAGE_CHANNEL_COUNT = 4
    MEAN_PIXEL = 4


class SunRGBDDataset(utils.Dataset):
    """Generates the sun rgbd dataset.
    """

    def load_sun_rgbd(self, dataset_dir, subset):
        """Load a subset of the sun rgbd dataset.
        Uses the rgb and the depth images
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("sunrgbd", 1, "bed")
        self.add_class("sunrgbd", 2, "books")
        self.add_class("sunrgbd", 3, "ceiling")
        self.add_class("sunrgbd", 4, "chair")
        self.add_class("sunrgbd", 5, "floor")
        self.add_class("sunrgbd", 6, "furniture")
        self.add_class("sunrgbd", 7, "objects")
        self.add_class("sunrgbd", 8, "picture")
        self.add_class("sunrgbd", 9, "sofa")
        self.add_class("sunrgbd", 10, "table")
        self.add_class("sunrgbd", 11, "tv")
        self.add_class("sunrgbd", 12, "wall")
        self.add_class("sunrgbd", 13, "window")

        # Train or validation dataset?
        assert subset in ["train13", "test13", "split/test13", "split/val13"]
        dataset_file = os.path.join(dataset_dir, subset + ".txt")

        # Load annotations
        f = open(dataset_file, "r")
        annotations = list(f)
        f.close()

        # Add images
        for a in annotations:
            files = a.split(" ")
            rgb_image = files[0]
            dpt_image = files[1]
            lbl_image = files[2]

            rgb_image_path = os.path.join(dataset_dir, rgb_image)
            dpt_image_path = os.path.join(dataset_dir, dpt_image)
            lbl_image_path = os.path.join(dataset_dir, lbl_image)
            dpt_image_path = dpt_image_path.strip()
            rgb_image_path = rgb_image_path.strip()
            lbl_image_path = lbl_image_path.strip()
            msk_full_path = lbl_image_path + ".mask.npy"
            cls_full_path = lbl_image_path + ".class_ids.npy"

            self.add_image(
                "sunrgbd",
                image_id=rgb_image,  # use file name as a unique image id
                path=rgb_image_path,
                dpt_image_path=dpt_image_path,
                lbl_image_path=lbl_image_path,
                msk_full_path=msk_full_path,
                cls_full_path=cls_full_path)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "sunrgbd":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a sun dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "sunrgbd":
            return super(self.__class__, self).load_mask(image_id)
        masks = np.load(image_info["msk_full_path"])
        class_ids = np.load(image_info["cls_full_path"])
        return masks, class_ids

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,4] Numpy array.
        an image consists of 3 color channels and 1 depth channel
        """
        # Load image
        rgb_image = super().load_image(image_id)
        dpt_image = skimage.io.imread(self.image_info[image_id]['dpt_image_path'])
        width = rgb_image.shape[0]
        height = rgb_image.shape[1]
        image = np.zeros([width, height, 4], np.uint8)

        image[:, :, 0] = rgb_image[:, :, 0]
        image[:, :, 1] = rgb_image[:, :, 1]
        image[:, :, 2] = rgb_image[:, :, 2]
        image[:, :, 3] = (dpt_image / 65535 * 255).astype(np.uint8)  # normalized conversion from uint16 to uint8

        return image
