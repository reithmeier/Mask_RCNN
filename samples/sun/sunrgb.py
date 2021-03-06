# **********************************************************************************************************************
#
# brief:    Mask R-CNN
#           Configurations and data loading code for the sun rgbd dataset.
#
# author:   Lukas Reithmeier
# date:     19.04.2020
#
# **********************************************************************************************************************

import os
import sys
import math
import random
import numpy as np
import cv2
import skimage.draw
import skimage.io

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils


class SunRGBConfig(Config):
    """Configuration for training on the sun rgbd dataset.
    Derives from the base Config class and overrides values specific
    to the sun rgbd dataset.
    """
    # Give the configuration a recognizable name
    NAME = "sunrgb"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 13  # background + 13 classes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # smaller anchors, since images are 512x512
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 50

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 500

    # use small validation steps since the epoch is small
    # VALIDATION_STEPS = 5

    # Skip detections with < 90% confidence
    # DETECTION_MIN_CONFIDENCE = 0.9


class SunRGBDataset(utils.Dataset):
    """Generates the sun rgbd dataset.
    Only uses the RGB Images
    """

    def load_sun_rgb(self, dataset_dir, subset):
        """Load a subset of the sun rgbd dataset.
        Only use the rgb images
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("sunrgb", 1, "bed")
        self.add_class("sunrgb", 2, "books")
        self.add_class("sunrgb", 3, "ceiling")
        self.add_class("sunrgb", 4, "chair")
        self.add_class("sunrgb", 5, "floor")
        self.add_class("sunrgb", 6, "furniture")
        self.add_class("sunrgb", 7, "objects")
        self.add_class("sunrgb", 8, "picture")
        self.add_class("sunrgb", 9, "sofa")
        self.add_class("sunrgb", 10, "table")
        self.add_class("sunrgb", 11, "tv")
        self.add_class("sunrgb", 12, "wall")
        self.add_class("sunrgb", 13, "window")

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
            lbl_image = files[2]

            rgb_image_path = os.path.join(dataset_dir, rgb_image)
            lbl_image_path = os.path.join(dataset_dir, lbl_image)
            rgb_image_path = rgb_image_path.strip()
            lbl_image_path = lbl_image_path.strip()
            msk_full_path = lbl_image_path + ".mask.npy"
            cls_full_path = lbl_image_path + ".class_ids.npy"

            self.add_image(
                "sunrgb",
                image_id=rgb_image,  # use file name as a unique image id
                path=rgb_image_path,
                lbl_image_path=lbl_image_path,
                msk_full_path=msk_full_path,
                cls_full_path=cls_full_path,
                #rgb_image=self.open_image(rgb_image_path),
                #masks=self.open_mask(msk_full_path),
                #class_ids=self.open_class_ids(cls_full_path)
            )

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "sunrgb":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

    def open_mask(self, path):
        return np.load(path)

    def open_class_ids(self, path):
        return np.load(path)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a sun dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "sunrgb":
            return super(self.__class__, self).load_mask(image_id)

        #return self.image_info[image_id]['masks'], self.image_info[image_id]['class_ids']
        return self.open_mask(self.image_info[image_id]['msk_full_path']), self.open_class_ids(self.image_info[image_id]['cls_full_path'])

    def open_image(self, image_path):
        # Load image
        image = skimage.io.imread(image_path)
        return image

    def load_image(self, image_id):
        """access in memory image
        """
        # If not a sun dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "sunrgb":
            return super(self.__class__, self).load_mask(image_id)

        return self.open_image(self.image_info[image_id]['path'])
        #return self.image_info[image_id]['rgb_image']
