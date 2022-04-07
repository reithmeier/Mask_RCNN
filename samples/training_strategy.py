# **********************************************************************************************************************
#
# brief:    script to perform a single training run
#
# author:   Lukas Reithmeier
# date:     12.05.2020
#
# **********************************************************************************************************************


import argparse
import os
import sys

import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np

from samples.sun.sund3 import SunD3Config, SunD3Dataset
from samples.sun.sunrgb import SunRGBConfig, SunRGBDataset
from samples.sun.sunrgbd import SunRGBDConfig, SunRGBDDataset
from samples.sun.sunrgbd_parallel import SunRGBDParallelConfig, SunRGBDParallelDataset
from samples.sun.sunrgbd_fusenet import SunRGBDFusenetConfig, SunRGBDFusenetDataset

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append('.')
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
import keras
from samples.elevator.elevator_d3 import ElevatorD3Config, ElevatorD3Dataset
from samples.elevator.elevator_rgb import ElevatorRGBConfig, ElevatorRGBDataset
from samples.elevator.elevator_rgbd import ElevatorRGBDConfig, ElevatorRGBDDataset
from samples.elevator.elevator_rgbd_parallel import ElevatorRGBDParallelConfig, ElevatorRGBDParallelDataset
from samples.elevator.elevator_rgbd_fusenet import ElevatorRGBDFusenetConfig, ElevatorRGBDFusenetDataset


def train_model(config, dataset_train, dataset_val, epochs, model_dir, augment, train_layers, load_model, model_name,
                init_epoch):
    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=model_dir)
    print(model.keras_model.summary())
    if load_model:
        model.load_weights(model_name, by_name=True)  # ,
        model.epoch = init_epoch

    augmentation = iaa.Sequential([
        iaa.Fliplr(0.5),  # horizontally flip 50% of the images
        iaa.Flipud(0.5),  # horizontally flip 50% of the images
        iaa.Sometimes(0.5, iaa.CropAndPad(
            percent=(-0.05, 0.0),
            pad_mode=ia.ALL,
            pad_cval=(0, 255)
        )),
        iaa.Sometimes(0.5, iaa.Affine(
            scale={"x": (0.8, 1.0), "y": (0.8, 1.0)},
            # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent (per axis)
            rotate=(-45, 45),  # rotate by -45 to +45 degrees
            shear=(-16, 16),  # shear by -16 to +16 degrees
            order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
    ])

    if train_layers == "heads_4+_all":
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

    elif train_layers == "all_4+_heads":
        # training stage 1
        print("Train all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE * 5,
                    epochs=epochs[0],
                    layers='all',
                    augmentation=augmentation)
        # training stage 2
        print("Fine tune ResNet layers 4+")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=epochs[1],
                    layers='4+',
                    augmentation=augmentation)

        # training stage 3
        print("Train head layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 5,
                    epochs=epochs[2],
                    layers='heads',
                    augmentation=augmentation)

    elif train_layers == "all":
        custom_callbacks = [keras.callbacks.LearningRateScheduler(
            lambda epoch_index: config.LEARNING_RATE if epoch_index < epochs[0]
            else config.LEARNING_RATE / 5 if epoch_index < epochs[1]
            else config.LEARNING_RATE / 10
        )]
        # training stage 1
        print("Train all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE * 5,
                    epochs=epochs[2],
                    layers='all',
                    augmentation=augmentation,
                    custom_callbacks=custom_callbacks)
    return model


def inference_calculation(config, model_dir, model_path, dataset_val):
    config.BATCH_SIZE = 1
    config.IMAGES_PER_GPU = 1
    model = modellib.MaskRCNN(mode="inference",
                              config=config,
                              model_dir=model_dir)
    model.load_weights(model_path, by_name=True)

    mean_average_precision, f1_score = calc_mean_average_precision(dataset_val=dataset_val, inference_config=config,
                                                                   model=model)
    print("mAP: ", mean_average_precision)
    print("F1s: ", f1_score)
    return mean_average_precision, f1_score


def calc_mean_average_precision(dataset_val, inference_config, model):
    # Compute VOC-Style mAP @ IoU=0.5
    # Running on 10 images. Increase for better accuracy.
    image_ids = dataset_val.image_ids
    APs = []
    F1s = []
    for image_id in image_ids:
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset_val, inference_config,
                                   image_id, use_mini_mask=False)

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


def main(data_set, strategy, data_dir, model_dir, augment, load_model, model_name, init_epoch, train_layers, backbone,
         batch_size):
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

    config.BACKBONE = backbone
    config.BATCH_SIZE = batch_size
    config.IMAGES_PER_GPU = batch_size
    #config.OPTIMIZER = "ADAM"
    #config.LEARNING_RATE = 0.0001
    # config.LEARNING_RATE = 0.0001
    config.DETECTION_MIN_CONFIDENCE = 0.8
    config.TRAIN_ROIS_PER_IMAGE = 50
    # config.IMAGE_MIN_DIM = 256
    # config.IMAGE_MAX_DIM = 256
    # config.DROPOUT_RATE = -1
    # config.RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    config.NUM_FILTERS = [32, 32, 64, 128, 256]
    config.display()

    epochs = [150, 300, 300]
    model_path = model_dir + data_set + "_" + strategy + ".h5"
    model = train_model(config=config, dataset_train=dataset_train, dataset_val=dataset_val, epochs=epochs,
                        model_dir=model_dir, augment=augment, load_model=load_model, model_name=model_name,
                        init_epoch=init_epoch, train_layers=train_layers)
    model.keras_model.save_weights(model_path)

    inference_calculation(config=config, model_dir=model_dir, model_path=model_path, dataset_val=dataset_val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, help="Data directory",
                        default=os.path.abspath(
                            "../datasets/SUN_RGBD/crop"))
    parser.add_argument("-m", "--model_dir", type=str, help="Directory to store weights and results",
                        default="../datasets/logs")
    parser.add_argument("-s", "--strategy", type=str, help="[D3, RGB, RGBD, RGBDParallel, RGBDFusenet]",
                        default="RGBDFusenet")
    parser.add_argument("-w", "--data_set", type=str, help="[SUN, ELEVATOR]", default="SUN")
    args = parser.parse_args()

    main(data_set=args.data_set, strategy=args.strategy, data_dir=args.data_dir, model_dir=args.model_dir, augment=True,
         load_model=False,
         model_name="./mask_rcnn_sunrgb_0282.h5",
         init_epoch=282, train_layers="all",
         backbone="fusenet", batch_size=2)
