import argparse
import multiprocessing
import os
import sys
from collections import OrderedDict

import imgaug.augmenters as iaa
import joblib
import numpy as np
import tensorflow as tf
from hyperopt import fmin, tpe, space_eval
from hyperopt import hp, Trials
from tensorboard.plugins.hparams import api as tbhp

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append('.')
sys.path.append(ROOT_DIR)  # To find local version of the library
import mrcnn.model as modellib
from mrcnn import utils
from samples.sun.sund3 import SunD3Config, SunD3Dataset

from samples.sun.sund3 import SunD3Config, SunD3Dataset
from samples.sun.sunrgb import SunRGBConfig, SunRGBDataset
from samples.sun.sunrgbd import SunRGBDConfig, SunRGBDDataset
from samples.sun.sunrgbd_parallel import SunRGBDParallelConfig, SunRGBDParallelDataset
from samples.sun.sunrgbd_fusenet import SunRGBDFusenetConfig, SunRGBDFusenetDataset
from samples.elevator.elevator_d3 import ElevatorD3Config, ElevatorD3Dataset
from samples.elevator.elevator_rgb import ElevatorRGBConfig, ElevatorRGBDataset
from samples.elevator.elevator_rgbd import ElevatorRGBDConfig, ElevatorRGBDDataset
from samples.elevator.elevator_rgbd_parallel import ElevatorRGBDParallelConfig, ElevatorRGBDParallelDataset
from samples.elevator.elevator_rgbd_fusenet import ElevatorRGBDFusenetConfig, ElevatorRGBDFusenetDataset


def inference_calculation(config, model_dir, model_path, dataset_val):
    """
    calculate mean average precision and f1 score with the inference model
    :param config: inference config
    :param model_dir: directory to log the model
    :param model_path: path of weights file
    :param dataset_val: validation data set
    :return: mean average precision, f1 score
    """
    model = modellib.MaskRCNN(mode="inference",
                              config=config,
                              model_dir=model_dir, unique_log_name=False)
    model.load_weights(model_path, by_name=True)

    mean_average_precision, f1_score = evaluate(dataset_val=dataset_val, inference_config=config,
                                                model=model)
    print("mAP: ", mean_average_precision)
    print("F1s: ", f1_score)
    return mean_average_precision, f1_score


def evaluate(dataset_val, inference_config, model):
    """
    calculate the mean average precision and the f1 score of a model
    :param dataset_val: validation data
    :param inference_config: inference configuration
    :param model: model
    :return: mean average precision, the f1 score
    """
    # Compute VOC-Style mAP @ IoU=0.5
    # Running on 10 images. Increase for better accuracy.
    image_ids = dataset_val.image_ids
    print(len(dataset_val.image_ids))
    APs = []
    F1s = []
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
        precision = np.mean(precisions)
        recall = np.mean(recalls)
        F1_instance = 2 * (precision * recall) / (precision + recall)
        APs.append(AP)
        F1s.append(F1_instance)
    print(len(APs), len(F1s))
    return np.mean(APs), np.mean(F1s)


def main(data_dir, backbone, model_dir, model_name, data_set, strategy):
    if data_set == "ELEVATOR":
        if strategy == "D3":
            config = ElevatorD3Config()
            dataset_train = ElevatorD3Dataset()
            dataset_train.load_elevator_d3(data_dir, "train")
            dataset_train.prepare()
            dataset_val = ElevatorD3Dataset()
            dataset_val.load_elevator_d3(data_dir, "validation")
            dataset_val.prepare()
        elif strategy == "RGBD":
            config = ElevatorRGBDConfig()
            dataset_train = ElevatorRGBDDataset()
            dataset_train.load_elevator_rgbd(data_dir, "train")
            dataset_train.prepare()
            dataset_val = ElevatorRGBDDataset()
            dataset_val.load_elevator_rgbd(data_dir, "validation")
            dataset_val.prepare()
        elif strategy == "RGBDParallel":
            config = ElevatorRGBDParallelConfig()
            dataset_train = ElevatorRGBDParallelDataset()
            dataset_train.load_elevator_rgbd_parallel(data_dir, "train")
            dataset_train.prepare()
            dataset_val = ElevatorRGBDParallelDataset()
            dataset_val.load_elevator_rgbd_parallel(data_dir, "validation")
            dataset_val.prepare()
        elif strategy == "RGBDFusenet":
            config = ElevatorRGBDFusenetConfig()
            dataset_train = ElevatorRGBDFusenetDataset()
            dataset_train.load_elevator_rgbd_fusenet(data_dir, "train")
            dataset_train.prepare()
            dataset_val = ElevatorRGBDFusenetDataset()
            dataset_val.load_elevator_rgbd_fusenet(data_dir, "validation")
            dataset_val.prepare()
        else:
            config = ElevatorRGBConfig()
            dataset_train = ElevatorRGBDataset()
            dataset_train.load_elevator_rgb(data_dir, "train")
            dataset_train.prepare()
            dataset_val = ElevatorRGBDataset()
            dataset_val.load_elevator_rgb(data_dir, "validation")
            dataset_val.prepare()
    else:
        if strategy == "D3":
            config = SunD3Config()
            dataset_train = SunD3Dataset()
            dataset_train.load_sun_d3(data_dir, "train13")
            dataset_train.prepare()
            dataset_val = SunD3Dataset()
            dataset_val.load_sun_d3(data_dir, "split/val13")
            dataset_val.prepare()
        elif strategy == "RGBD":
            config = SunRGBDConfig()
            dataset_train = SunRGBDDataset()
            dataset_train.load_sun_rgbd(data_dir, "train13")
            dataset_train.prepare()
            dataset_val = SunRGBDDataset()
            dataset_val.load_sun_rgbd(data_dir, "split/val13")
            dataset_val.prepare()
        elif strategy == "RGBDParallel":
            config = SunRGBDParallelConfig()
            dataset_train = SunRGBDParallelDataset()
            dataset_train.load_sun_rgbd_parallel(data_dir, "train13")
            dataset_train.prepare()
            dataset_val = SunRGBDParallelDataset()
            dataset_val.load_sun_rgbd_parallel(data_dir, "split/val13")
            dataset_val.prepare()
        elif strategy == "RGBDFusenet":
            config = SunRGBDFusenetConfig()
            dataset_train = SunRGBDFusenetDataset()
            dataset_train.load_sun_rgbd_fusenet(data_dir, "train13")
            dataset_train.prepare()
            dataset_val = SunRGBDFusenetDataset()
            dataset_val.load_sun_rgbd_fusenet(data_dir, "split/val13")
            dataset_val.prepare()
        else:
            config = SunRGBConfig()
            dataset_train = SunRGBDataset()
            dataset_train.load_sun_rgb(data_dir, "train13")
            dataset_train.prepare()
            dataset_val = SunRGBDataset()
            dataset_val.load_sun_rgb(data_dir, "split/val13")
            dataset_val.prepare()
    print("Image Count: {}".format(len(dataset_val.image_ids)))
    print("Class Count: {}".format(dataset_val.num_classes))
    for i, info in enumerate(dataset_val.class_info):
        print("{:3}. {:50}".format(i, info['name']))

    config.BACKBONE = backbone
    config.BATCH_SIZE = 1
    config.IMAGES_PER_GPU = 1
    # config.OPTIMIZER = "ADAM"
    # config.LEARNING_RATE = 0.0001
    config.DETECTION_MIN_CONFIDENCE = 0.7
    config.TRAIN_ROIS_PER_IMAGE = 100
    #config.IMAGE_MIN_DIM = 256
    #config.IMAGE_MAX_DIM = 256
    config.DROPOUT_RATE = -1
    #config.RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    config.display()

    model = modellib.MaskRCNN(mode="inference", config=config,
                              model_dir=model_dir)
    print(model.keras_model.summary())
    model.load_weights(model_name, by_name=True)  # ,
    # exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
    #         "mrcnn_bbox", "mrcnn_mask"])
    model.epoch = 300
    mean_average_precision, f1_score = evaluate(dataset_val=dataset_val, inference_config=config, model=model)
    print("mAP: ", mean_average_precision)
    print("F1s: ", f1_score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, help="Data directory",
                        default=os.path.abspath(
                            "C:\\public\\master_thesis_reithmeier_lukas\\sunrgbd\\SUN_RGBD\\crop"))
    parser.add_argument("-m", "--model_dir", type=str, help="Directory to store weights and results",
                        default="C:\\public\\master_thesis_reithmeier_lukas\\Mask_RCNN\\logs\\")
    parser.add_argument("-s", "--strategy", type=str, help="[D3, RGB, RGBD, RGBDParallel, RGBDFusenet]",
                        default="RGB")
    parser.add_argument("-w", "--data_set", type=str, help="[SUN, ELEVATOR]", default="SUN")
    args = parser.parse_args()

    main(data_dir=args.data_dir, model_dir=args.model_dir,
         model_name="C:\\public\\master_thesis_reithmeier_lukas\\Mask_RCNN\\logs\\sunrgb20200706T1041\\mask_rcnn_sunrgb_0282.h5",
         backbone="resnet50", strategy=args.strategy, data_set=args.data_set)
