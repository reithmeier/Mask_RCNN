import argparse
import os
import sys

import imgaug.augmenters as iaa
import numpy as np

from samples.sun.sunrgbd import SunRGBDConfig, SunRGBDDataset

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append('.')
sys.path.append(ROOT_DIR)  # To find local version of the library
import mrcnn.model as modellib
from mrcnn import utils

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

# hyper parameters

HP_BACKBONE = hp.HParam('backbone', hp.Discrete(["resnet50_batch_size1", "resnet50_batch_size2", "resnet101"]))
HP_TRAIN_ROIS_PER_IMAGE = hp.HParam('train_rois_per_image', hp.Discrete([50, 100, 200]))
HP_DETECTION_MIN_CONFIDENCE = hp.HParam('detection_min_confidence', hp.Discrete([0.6, 0.7, 0.8]))

METRIC_MAP = 'mean average precision'
METRIC_F1S = 'f1 score'

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
    hp.hparams_config(
        hparams=[HP_BACKBONE, HP_TRAIN_ROIS_PER_IMAGE, HP_DETECTION_MIN_CONFIDENCE],
        metrics=[hp.Metric(METRIC_MAP, display_name='mean average precision'),
                 hp.Metric(METRIC_F1S, display_name='f1 score')],
    )


def inference_calculation(config, model_dir, model_path, dataset_val):
    model = modellib.MaskRCNN(mode="inference",
                              config=config,
                              model_dir=model_dir, unique_log_name=False)
    model.load_weights(model_path, by_name=True)

    mean_average_precision, f1_score = calc_mean_average_precision(dataset_val=dataset_val, inference_config=config,
                                                                   model=model)
    print("mAP: ", mean_average_precision)
    print("F1s: ", f1_score)
    return mean_average_precision, f1_score


def calc_mean_average_precision(dataset_val, inference_config, model):
    # Compute VOC-Style mAP @ IoU=0.5
    # Running on 10 images. Increase for better accuracy.
    image_ids = np.random.choice(dataset_val.image_ids, 100)
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
    return np.mean(APs), np.mean(F1s)


def train_test_model(hparams, data_dir, log_dir, run_name):
    config = SunRGBDConfig()
    dataset_train = SunRGBDDataset()
    dataset_train.load_sun_rgbd(data_dir, "train13")
    dataset_train.prepare()
    dataset_val = SunRGBDDataset()
    dataset_val.load_sun_rgbd(data_dir, "split/val13")
    dataset_val.prepare()

    # Create model in training mode
    if hparams[HP_BACKBONE] == "resnet50_batch_size1":
        backbone = "resnet50"
        batch_size = 1
    elif hparams[HP_BACKBONE] == "resnet50_batch_size2":
        backbone = "resnet50"
        batch_size = 2
    else:
        backbone = "resnet101"
        batch_size = 1

    config.BACKBONE = backbone
    config.BATCH_SIZE = batch_size
    config.IMAGES_PER_GPU = batch_size

    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=log_dir, unique_log_name=False)
    augmentation = iaa.Sequential([
        iaa.Fliplr(0.5)
    ])

    custom_callbacks = [hp.KerasCallback(log_dir, hparams)]

    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=1,
                layers='all',
                augmentation=augmentation,
                custom_callbacks=custom_callbacks)

    model_path = log_dir + run_name + ".h5"
    model.keras_model.save_weights(model_path)
    map, f1s = inference_calculation(config=config, model_path=model_path, model_dir=log_dir, dataset_val=dataset_val)

    return map, f1s


def run(hparams, data_dir, log_dir, run_name):
    with tf.summary.create_file_writer(log_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        map, f1s = train_test_model(hparams, data_dir=data_dir, log_dir=log_dir, run_name=run_name)
        tf.summary.scalar(METRIC_MAP, map, step=1)
        tf.summary.scalar(METRIC_F1S, f1s, step=1)


def main(data_dir, log_dir):
    session_num = 0

    for backbone in HP_BACKBONE.domain.values:
        for train_rois_per_image in HP_TRAIN_ROIS_PER_IMAGE.domain.values:
            for detection_min_confidence in HP_DETECTION_MIN_CONFIDENCE.domain.values:
                hparams = {
                    HP_BACKBONE: backbone,
                    HP_TRAIN_ROIS_PER_IMAGE: train_rois_per_image,
                    HP_DETECTION_MIN_CONFIDENCE: detection_min_confidence,
                }
                run_name = "run-%d" % session_num
                print('--- Starting trial: %s' % run_name)
                print({h.name: hparams[h] for h in hparams})
                run_dir = log_dir + run_name
                run(hparams, data_dir=data_dir, log_dir=run_dir, run_name=run_name)
                session_num += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, help="Data directory",
                        default=os.path.abspath(
                            "C:\public\master_thesis_reithmeier_lukas\sunrgbd\SUN_RGBD\crop"))  # os.path.abspath("I:\Data\elevator\preprocessed"))
    parser.add_argument("-m", "--model_dir", type=str, help="Directory to store weights and results",
                        default=ROOT_DIR + "/logs/")
    args = parser.parse_args()

    main(data_dir=args.data_dir, log_dir=args.model_dir)
