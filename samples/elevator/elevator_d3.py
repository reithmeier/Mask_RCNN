"""
Mask R-CNN
Configurations and data loading code for the elevator dataset.
uses depth data only
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


class ElevatorD3Config(ElevatorRGBConfig):
    """Configuration for training on the elevator dataset.
    Derives from the base Config class and overrides values specific
    to the elevator dataset.
    """
    # Give the configuration a recognizable name
    NAME = "elevator_d3"


class ElevatorD3Dataset(utils.Dataset):
    """Generates the elevator dataset.
    uses depth data only
    """

    def __init__(self, class_map=None):
        self.class_name_to_id = {}
        super(ElevatorD3Dataset, self).__init__(class_map)

    def add_class(self, source, class_id, class_name):
        self.class_name_to_id[class_name] = class_id
        super(ElevatorD3Dataset, self).add_class(source=source, class_id=class_id, class_name=class_name)

    def load_elevator_d3(self, dataset_dir, subset):
        """Load a subset of the elevator dataset.
        Only use the depth images
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("elevator_d3", 1, "human_standing")
        self.add_class("elevator_d3", 2, "human_sitting")
        self.add_class("elevator_d3", 3, "human_lying")
        self.add_class("elevator_d3", 4, "bag")
        self.add_class("elevator_d3", 5, "box")
        self.add_class("elevator_d3", 6, "crate")
        self.add_class("elevator_d3", 7, "plant")
        self.add_class("elevator_d3", 8, "chair")
        self.add_class("elevator_d3", 9, "object")
        self.add_class("elevator_d3", 10, "door")

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

            # rgb_file = files[0] # not used here
            dpt_file = files[1]
            lbl_file = files[4]

            dpt_full_path = os.path.join(dataset_dir, dpt_file)
            lbl_full_path = os.path.join(dataset_dir, lbl_file)
            dpt_full_path = dpt_full_path.strip()
            lbl_full_path = lbl_full_path.strip()

            self.add_image(
                "elevator_d3",
                image_id=dpt_file,  # use file name as a unique image id
                path=dpt_full_path,
                lbl_full_path=lbl_full_path)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "elevator_d3":
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
        if image_info["source"] != "elevator_d3":
            return super(self.__class__, self).load_mask(image_id)

        return util.create_mask(lbl_full_path=self.image_info[image_id]["lbl_full_path"],
                                class_name_to_id=self.class_name_to_id)

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
