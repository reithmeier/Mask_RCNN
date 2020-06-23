# **********************************************************************************************************************
#
# brief:    Mask R-CNN
#           Configurations and data loading code for the sun rgbd dataset.
#           uses depth data only
#           see: https://github.com/matterport/Mask_RCNN/wiki#training-with-rgb-d-or-grayscale-images
#
# author:   Lukas Reithmeier
# date:     20.04.2020
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


class SunD3Config(SunRGBConfig):
    """Configuration for training on the sun rgbd dataset.
    Derives from the base Config class and overrides values specific
    to the sun rgbd dataset.
    """
    # Give the configuration a recognizable name
    NAME = "sund3"


class SunD3Dataset(utils.Dataset):
    """Generates the sun rgbd dataset.
    uses depth data only
    """

    def load_sun_d3(self, dataset_dir, subset):
        """Load a subset of the sun rgbd dataset.
        Only use the depth images
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("sund3", 1, "bed")
        self.add_class("sund3", 2, "books")
        self.add_class("sund3", 3, "ceiling")
        self.add_class("sund3", 4, "chair")
        self.add_class("sund3", 5, "floor")
        self.add_class("sund3", 6, "furniture")
        self.add_class("sund3", 7, "objects")
        self.add_class("sund3", 8, "picture")
        self.add_class("sund3", 9, "sofa")
        self.add_class("sund3", 10, "table")
        self.add_class("sund3", 11, "tv")
        self.add_class("sund3", 12, "wall")
        self.add_class("sund3", 13, "window")

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
            dpt_image = files[1]
            lbl_image = files[2]

            dpt_image_path = os.path.join(dataset_dir, dpt_image)
            lbl_image_path = os.path.join(dataset_dir, lbl_image)
            dpt_image_path = dpt_image_path.strip()
            lbl_image_path = lbl_image_path.strip()
            msk_full_path = lbl_image_path + ".mask.npy"
            cls_full_path = lbl_image_path + ".class_ids.npy"

            self.add_image(
                "sund3",
                image_id=dpt_image_path,  # use file name as a unique image id
                path=dpt_image_path,
                lbl_image_path=lbl_image_path,
                msk_full_path=msk_full_path,
                cls_full_path=cls_full_path)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "sund3":
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
        if image_info["source"] != "sund3":
            return super(self.__class__, self).load_mask(image_id)
        masks = np.load(image_info["msk_full_path"])
        class_ids = np.load(image_info["cls_full_path"])
        return masks, class_ids

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        overrides load_image, since depth image is uint16 and not uint8
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb((image / 65535 * 255).astype(np.uint8))
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image
