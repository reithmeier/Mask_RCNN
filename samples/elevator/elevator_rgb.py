"""
Mask R-CNN
Configurations and data loading code for the elevator dataset.
"""

import json
import os
import sys

import numpy as np
import skimage.draw

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

    @staticmethod
    def draw_polygon(polygon, width, height):
        y_values = [p[0] * width / 100 for p in polygon]
        x_values = [p[1] * height / 100 for p in polygon]
        rr, cc = skimage.draw.polygon(x_values, y_values)
        # labels might extend over boundaries, due to preprocessing
        rr = [height - 1 if r > height - 1 else r for r in rr]
        cc = [width - 1 if c > width - 1 else c for c in cc]
        rr = [0 if r < 0 else r for r in rr]
        cc = [0 if c < 0 else c for c in cc]
        return rr, cc

    @staticmethod
    def find_relation(relations, identifier):
        for relation in relations:
            if relation["from_id"] == identifier:
                return True, relation["to_id"]
            if relation["to_id"] == identifier:
                return True, relation["from_id"]
        return False, ""

    @staticmethod
    def find_result(results, identifier):
        for result in results:
            if result["id"] == identifier:
                return result["value"]["points"]
        return []

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

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        lbl_full_path = info["lbl_full_path"]
        with open(lbl_full_path) as lbl_file:
            labels = json.load(lbl_file)
        results = labels["completions"][-1]["result"]  # always get the latest entries
        if len(results) == 0:  # no labels
            return np.zeros([0, 0, 0], dtype=np.bool), np.array([])

        height = results[0]["original_height"]
        width = results[0]["original_width"]
        instance_count = len(results)

        mask = np.zeros([height, width, instance_count], np.bool)
        class_ids = np.zeros([instance_count], np.int)

        relations = []
        # relations indicate that two polygons belong to the same object instance in the image
        # therefore relations need to be found and masks need to be merged
        for result in results:
            if result["type"] != "relation":
                continue
            # keys: from_id, to_id
            relations.append(result)

        skip_ids = []  # skip already merged polygons

        i = 0
        for result in results:
            # skip non-polygons
            if result["type"] != "polygonlabels":
                continue
            if result["id"] in skip_ids:
                continue

            rr, cc = self.draw_polygon(result["value"]["points"], width, height)

            mask[rr, cc, i] = True

            # find relation
            has_relation, relation_id = self.find_relation(relations, result["id"])
            if has_relation:
                rel_rr, rel_cc = self.draw_polygon(self.find_result(results, relation_id), width, height)
                mask[rel_rr, rel_cc, i] = True
                skip_ids.append(relation_id)

            label_txt = result["value"]["polygonlabels"][0]
            class_ids[i] = self.class_name_to_id[label_txt]
            i = i + 1

        return mask, class_ids
