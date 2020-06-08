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


def run(hparams, data_dir, log_dir, run_name):
    with tf.summary.create_file_writer(log_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        train_test_model(hparams, data_dir=data_dir, log_dir=log_dir, run_name=run_name)


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
