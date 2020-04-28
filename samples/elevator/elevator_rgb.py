"""
Mask R-CNN
Configurations and data loading code for the elevator dataset.
"""

import os
import sys

from . import util

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils


class ElevatorRGBConfig(Config):
    """Configuration for training on the elevator dataset.
    Derives from the base Config class and overrides values specific
    to the elevator dataset.
    """
    # Give the configuration a recognizable name
    NAME = "elevator_rgb"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 10  # background + 10 classes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # smaller anchors, since images are 512x512
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    # TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    # VALIDATION_STEPS = 5

    # Skip detections with < 90% confidence
    # DETECTION_MIN_CONFIDENCE = 0.9


class ElevatorRGBDataset(utils.Dataset):
    """Generates the elevator dataset.
    Only uses the RGB Images
    """

    def __init__(self, class_map=None):
        self.class_name_to_id = {}
        super(ElevatorRGBDataset, self).__init__(class_map)

    def add_class(self, source, class_id, class_name):
        self.class_name_to_id[class_name] = class_id
        super(ElevatorRGBDataset, self).add_class(source=source, class_id=class_id, class_name=class_name)

    def load_elevator_rgb(self, dataset_dir, subset):
        """Load a subset of the elevator dataset.
        Only use the rgb images
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train, val or test
        """
        # Add classes. We have only one class to add.
        self.add_class("elevator_rgb", 1, "human_standing")
        self.add_class("elevator_rgb", 2, "human_sitting")
        self.add_class("elevator_rgb", 3, "human_lying")
        self.add_class("elevator_rgb", 4, "bag")
        self.add_class("elevator_rgb", 5, "box")
        self.add_class("elevator_rgb", 6, "crate")
        self.add_class("elevator_rgb", 7, "plant")
        self.add_class("elevator_rgb", 8, "chair")
        self.add_class("elevator_rgb", 9, "object")
        self.add_class("elevator_rgb", 10, "human_other")

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
            # dpt_file = files[1]  # not used here
            lbl_file = files[4]

            rgb_full_path = os.path.join(dataset_dir, rgb_file)
            lbl_full_path = os.path.join(dataset_dir, lbl_file)
            rgb_full_path = rgb_full_path.strip()
            lbl_full_path = lbl_full_path.strip()

            self.add_image(
                "elevator_rgb",
                image_id=rgb_file,  # use file name as a unique image id
                path=rgb_full_path,
                lbl_full_path=lbl_full_path)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "elevator_rgb":
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
        if image_info["source"] != "elevator_rgb":
            return super(self.__class__, self).load_mask(image_id)

        return util.create_mask(lbl_full_path=self.image_info[image_id]["lbl_full_path"],
                                class_name_to_id=self.class_name_to_id)
