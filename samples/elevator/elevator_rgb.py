"""
Mask R-CNN
Configurations and data loading code for the elevator dataset.
"""

import argparse
import copy
import os
import random
import sys

import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append('.')
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
from mrcnn.model import MaskRCNN


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

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    # VALIDATION_STEPS = 5

    # Skip detections with < 90% confidence
    # DETECTION_MIN_CONFIDENCE = 0.9

    # Maximum number of ground truth instances to use in one image
    # there are only few objects per image in the elevator dataset
    # see: class_distribution.py
    MAX_GT_INSTANCES = 16


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
            msk_full_path = lbl_full_path + ".mask.npy"
            cls_full_path = lbl_full_path + ".class_ids.npy"
            self.add_image(
                "elevator_rgb",
                image_id=rgb_file,  # use file name as a unique image id
                path=rgb_full_path,
                lbl_full_path=lbl_full_path,
                msk_full_path=msk_full_path,
                cls_full_path=cls_full_path
            )

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

        mask = np.load(image_info["msk_full_path"])
        class_ids = np.load(image_info["cls_full_path"])
        return mask, class_ids



def run_training():
    config = ElevatorRGBConfig()
    config.display()

    # Training dataset
    dataset_train = ElevatorRGBDataset()
    dataset_train.load_elevator_rgb(args.data_dir, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = ElevatorRGBDataset()
    dataset_val.load_elevator_rgb(args.data_dir, "validation")
    dataset_val.prepare()

    print("Image Count: {}".format(len(dataset_train.image_ids)))
    print("Class Count: {}".format(dataset_train.num_classes))
    for i, info in enumerate(dataset_train.class_info):
        print("{:3}. {:50}".format(i, info['name']))

    print("Image Count: {}".format(len(dataset_val.image_ids)))
    print("Class Count: {}".format(dataset_val.num_classes))
    for i, info in enumerate(dataset_val.class_info):
        print("{:3}. {:50}".format(i, info['name']))

    # Create model in training mode
    model = MaskRCNN(mode="training", config=config,
                     model_dir=args.model_dir)

    model.load_weights(args.coco_path, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])

    augmentation = iaa.Sequential([
        iaa.Sometimes(0.5, iaa.CropAndPad(
            percent=(-0.05, 0.1),
            pad_mode=ia.ALL,
            pad_cval=(0, 255)
        )),
        iaa.Sometimes(0.5, iaa.Affine(
            scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            rotate=(-30, 30),
            shear=(-8, 8),
            order=[0, 1],
            cval=(0, 255),
            mode=ia.ALL
        )),
        iaa.Fliplr(0.5)
    ])

    print(model.keras_model.summary())

    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=10,
                layers='heads',
                augmentation=augmentation)

    # Fine tune all layers
    # Passing layers="all" trains all layers. You can also
    # pass a regular expression to select which layers to
    # train by name pattern.
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=20,
                layers="all",
                augmentation=augmentation)

    # Save weights
    # Typically not needed because callbacks save after every epoch
    # Uncomment to save manually
    model_path = os.path.join(args.model_dir, args.output)
    print(model_path)
    model.keras_model.save_weights(model_path)


def calc_mean_average_precision(dataset_val, inference_config, model):
    # Compute VOC-Style mAP @ IoU=0.5
    # Running on 10 images. Increase for better accuracy.
    image_ids = np.random.choice(dataset_val.image_ids, 100)
    APs = []
    for image_id in image_ids:
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset_val, inference_config,
                                   image_id, use_mini_mask=False)
        molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
        # Compute AP
        AP, precisions, recalls, overlaps = \
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                             r["rois"], r["class_ids"], r["scores"], r['masks'])
        APs.append(AP)
    return np.mean(APs)


def run_inference():
    # Training dataset
    dataset_train = ElevatorRGBDataset()
    dataset_train.load_elevator_rgb(args.data_dir, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = ElevatorRGBDataset()
    dataset_val.load_elevator_rgb(args.data_dir, "validation")
    dataset_val.prepare()

    class InferenceConfig(ElevatorRGBConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    inference_config = InferenceConfig()

    # Recreate the model in inference mode
    model = MaskRCNN(mode="inference",
                     config=inference_config,
                     model_dir=args.model_dir)

    # Get path to saved weights
    # Either set a specific path or find last trained weights
    model_path = os.path.join(args.model_dir, "mask_rcnn_elevator_rgb.h5")

    # Load trained weights
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    # Test on a random image
    image_id = random.choice(dataset_val.image_ids)
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)

    log("original_image", original_image)
    log("image_meta", image_meta)
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)

    visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                                dataset_train.class_names, figsize=(8, 8))

    results = model.detect([original_image], verbose=1)

    r = results[0]
    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                                dataset_val.class_names, r['scores'])

    mean_average_precision = calc_mean_average_precision(dataset_val=dataset_val, inference_config=inference_config,
                                                         model=model)
    print("mAP: ", mean_average_precision)


def train_with_config(config, epochs, data_dir, model_dir):
    # Training dataset
    dataset_train = ElevatorRGBDataset()
    dataset_train.load_elevator_rgb(data_dir, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = ElevatorRGBDataset()
    dataset_val.load_elevator_rgb(data_dir, "validation")
    dataset_val.prepare()

    # Create model in training mode
    model = MaskRCNN(mode="training", config=config,
                     model_dir=model_dir)

    """
        iaa.Sometimes(0.5, iaa.CropAndPad(
            percent=(-0.05, 0.1),
            pad_mode=ia.ALL,
            pad_cval=(0, 255)
        )),
        iaa.Sometimes(0.5, iaa.Affine(
            scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            rotate=(-30, 30),
            shear=(-8, 8),
            order=[0, 1],
            cval=(0, 255),
            mode=ia.ALL
        )),
    """
    augmentation = iaa.Sequential([
        iaa.Fliplr(0.5)
    ])

    # training stage 1
    print("Train head layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=epochs[0],
                layers='heads',
                augmentation=augmentation)
    # training stage 2
    print("Fine tune ResNet layers 4+")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=epochs[1],
                layers='4+',
                augmentation=augmentation)

    # training stage 3
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=epochs[2],
                layers='all',
                augmentation=augmentation)

    model_path = model_dir + "elevator_rgb_" + \
                 str(config.TRAIN_ROIS_PER_IMAGE) + "_" + \
                 str(config.DETECTION_MIN_CONFIDENCE) + ".h5"

    model.keras_model.save_weights(model_path)

    inference_config = copy.deepcopy(config)
    inference_config.GPU_COUNT = 1
    inference_config.IMAGES_PER_GPU = 1
    inference_model = MaskRCNN(mode="inference",
                               config=inference_config,
                               model_dir=model_dir)
    inference_model.load_weights(model_path, by_name=True)

    mean_average_precision = calc_mean_average_precision(dataset_val=dataset_val, inference_config=inference_config,
                                                         model=inference_model)
    print("mAP: ", mean_average_precision)
    return mean_average_precision


def run_grid_search(data_dir, model_dir):
    """
    search for best configurations for parameters:
    TRAIN_ROIS_PER_IMAGE
    DETECTION_MIN_CONFIDENCE
    """
    train_rois_per_image = [32, 64, 128, 256, 512]
    detection_min_confidence = [0.5, 0.6, 0.7, 0.8, 0.9]
    config = ElevatorRGBConfig()

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    epochs = [20, 40, 80]

    result = np.zeros([len(train_rois_per_image), len(detection_min_confidence)])

    train_with_config(config=config, epochs=epochs, data_dir=data_dir, model_dir=model_dir)
    """
    i = 0
    j = 0
    for trpi in train_rois_per_image:
        for dmc in detection_min_confidence:
            print("train rois per image", trpi)
            print("detection min confidence", dmc)
            config.TRAIN_ROIS_PER_IMAGE = trpi
            config.DETECTION_MIN_CONFIDENCE = dmc
            result[i][j] = train_with_config(config=config, epochs=epochs, data_dir=data_dir, model_dir=model_dir)
            i = i + 1
            j = j + 1
        j = 0

    print(result)
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, help="Data directory",
                        default=os.path.abspath("I:\Data\elevator\preprocessed"))
    parser.add_argument("-m", "--model_dir", type=str, help="Input index file",
                        default=ROOT_DIR + "logs/")
    parser.add_argument("-c", "--coco_path", type=str, help="Path to pretrained coco weights",
                        default=ROOT_DIR + "mask_rcnn_coco.h5")
    parser.add_argument("-o", "--output", type=str, help="Name of output weights",
                        default="mask_rcnn_elevator_rgb.h5")
    parser.add_argument("-n", "--mode", type=str, help="[training, inference, grid_search]",
                        default="grid_search")
    args = parser.parse_args()
    if args.mode == "training":
        run_training()
    if args.mode == "inference":
        run_inference()
    if args.mode == "grid_search":
        run_grid_search(args.data_dir, args.model_dir)
