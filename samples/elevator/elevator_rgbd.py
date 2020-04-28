"""
Mask R-CNN
Configurations and data loading code for the elevator dataset.
uses rgbd data
@see https://github.com/matterport/Mask_RCNN/wiki#training-with-rgb-d-or-grayscale-images
"""

import os
import sys

import numpy as np
import skimage.draw

from . import util

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn import utils
from samples.elevator.elevator_rgb import ElevatorRGBConfig


class ElevatorRGBDConfig(ElevatorRGBConfig):
    """Configuration for training on the elevator dataset.
    Derives from the base Config class and overrides values specific
    to the elevator dataset.
    """
    # Give the configuration a recognizable name
    NAME = "elevator_rgbd"

    # 3 color channels +  1 depth channel
    IMAGE_CHANNEL_COUNT = 4
    MEAN_PIXEL = 4


class ElevatorRGBDDataset(utils.Dataset):
    """Generates the elevator dataset.
    """

    def __init__(self, class_map=None):
        self.class_name_to_id = {}
        super(ElevatorRGBDDataset, self).__init__(class_map)

    def add_class(self, source, class_id, class_name):
        self.class_name_to_id[class_name] = class_id
        super(ElevatorRGBDDataset, self).add_class(source=source, class_id=class_id, class_name=class_name)

    def load_elevator_rgbd(self, dataset_dir, subset):
        """Load a subset of the elevator dataset.
        Uses the rgb and the depth images
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("elevator_rgbd", 1, "human_standing")
        self.add_class("elevator_rgbd", 2, "human_sitting")
        self.add_class("elevator_rgbd", 3, "human_lying")
        self.add_class("elevator_rgbd", 4, "bag")
        self.add_class("elevator_rgbd", 5, "box")
        self.add_class("elevator_rgbd", 6, "crate")
        self.add_class("elevator_rgbd", 7, "plant")
        self.add_class("elevator_rgbd", 8, "chair")
        self.add_class("elevator_rgbd", 9, "object")
        self.add_class("elevator_rgbd", 10, "door")

        # Train or validation dataset?
        assert subset in ["train", "test", "validation"]
        dataset_file = os.path.join(dataset_dir, "split", subset + ".txt")

        # Load annotations
        f = open(dataset_file, "r")
        annotations = list(f)
        f.close()

        # Add images
        for a in annotations:
            files = a.split(" ")
            # file structure:
            # rgb dpt rgb_intrinsics dpt_intrinsics lbl

            rgb_file = files[0]
            dpt_file = files[1]
            lbl_file = files[4]

            rgb_full_path = os.path.join(dataset_dir, rgb_file)
            dpt_full_path = os.path.join(dataset_dir, dpt_file)
            lbl_full_path = os.path.join(dataset_dir, lbl_file)
            rgb_full_path = rgb_full_path.strip()
            dpt_full_path = dpt_full_path.strip()
            lbl_full_path = lbl_full_path.strip()

            self.add_image(
                "elevator_rgbd",
                image_id=rgb_file,  # use file name as a unique image id
                path=rgb_full_path,
                dpt_full_path=dpt_full_path,
                lbl_full_path=lbl_full_path)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "elevator_rgbd":
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
        # If not a elevator dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "elevator_rgbd":
            return super(self.__class__, self).load_mask(image_id)

        return util.create_mask(lbl_full_path=self.image_info[image_id]["lbl_full_path"],
                                class_name_to_id=self.class_name_to_id)

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,4] Numpy array.
        an image consists of 3 color channels and 1 depth channel
        """
        # Load image
        rgb_image = super().load_image(image_id)
        dpt_image = skimage.io.imread(self.image_info[image_id]['dpt_full_path'])
        width = rgb_image.shape[0]
        height = rgb_image.shape[1]
        image = np.zeros([width, height, 4], np.uint8)

        image[:, :, 0] = rgb_image[:, :, 0]
        image[:, :, 1] = rgb_image[:, :, 1]
        image[:, :, 2] = rgb_image[:, :, 2]
        image[:, :, 3] = (dpt_image / 65535 * 255).astype(np.uint8)  # normalized conversion from uint16 to uint8

        return image
