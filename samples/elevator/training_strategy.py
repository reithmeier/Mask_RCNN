import argparse
import os
import sys

import imgaug.augmenters as iaa
import numpy as np

from samples.sun.sund3 import SunD3Config, SunD3Dataset
from samples.sun.sunrgb import SunRGBConfig, SunRGBDataset
from samples.sun.sunrgbd import SunRGBDConfig, SunRGBDDataset
from samples.sun.sunrgbd_parallel import SunRGBDParallelConfig, SunRGBDParallelDataset


# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append('.')
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib

from samples.elevator.elevator_d3 import ElevatorD3Config, ElevatorD3Dataset
from samples.elevator.elevator_rgb import ElevatorRGBConfig, ElevatorRGBDataset
from samples.elevator.elevator_rgbd import ElevatorRGBDConfig, ElevatorRGBDDataset
from samples.elevator.elevator_rgbd_parallel import ElevatorRGBDParallelConfig, ElevatorRGBDParallelDataset


def train_model(config, dataset_train, dataset_val, epochs, model_dir, augment, load_model, model_name, init_epoch):
    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=model_dir)
    print(model.keras_model.summary())
    if load_model:
        model.keras_model.load_weights(model_dir + model_name)
        model.epoch = init_epoch
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
    if augment:
        augmentation = iaa.Sequential([
            iaa.Fliplr(0.5)
        ])
    else:
        augmentation = iaa.Sequential()

    ALTERNATIVE_TRAINING = False

    if not ALTERNATIVE_TRAINING:
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

    else:
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

    return model


def inference_calculation(config, model_dir, model_path, dataset_val):
    model = modellib.MaskRCNN(mode="inference",
                              config=config,
                              model_dir=model_dir)
    model.load_weights(model_path, by_name=True)

    mean_average_precision = calc_mean_average_precision(dataset_val=dataset_val, inference_config=config,
                                                         model=model)
    print("mAP: ", mean_average_precision)
    return mean_average_precision


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


def main(data_set, strategy, data_dir, model_dir, augment, load_model, model_name, init_epoch):
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
        else:
            config = SunRGBConfig()
            dataset_train = SunRGBDataset()
            dataset_train.load_sun_rgb(data_dir, "train13")
            dataset_train.prepare()
            dataset_val = SunRGBDataset()
            dataset_val.load_sun_rgb(data_dir, "split/val13")
            dataset_val.prepare()

    epochs = [20, 40, 80]
    model_path = model_dir + data_set + "_" + strategy + ".h5"
    model = train_model(config=config, dataset_train=dataset_train, dataset_val=dataset_val, epochs=epochs,
                        model_dir=model_dir, augment=augment, load_model=load_model, model_name=model_name,
                        init_epoch=init_epoch)
    model.keras_model.save_weights(model_path)

    inference_calculation(config=config, model_dir=model_dir, model_path=model_path, dataset_val=dataset_val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, help="Data directory",
                        default=os.path.abspath(
                            "I:\Data\elevator\preprocessed"))  # os.path.abspath("I:\Data\elevator\preprocessed"))
    parser.add_argument("-m", "--model_dir", type=str, help="Directory to store weights and results",
                        default=ROOT_DIR + "logs/")
    parser.add_argument("-s", "--strategy", type=str, help="[D3, RGB, RGBD, RGBDParallel]", default="RGBDParallel")
    parser.add_argument("-w", "--data_set", type=str, help="[SUN, ELEVATOR]", default="SUN")
    args = parser.parse_args()

    main(data_set=args.data_set, strategy=args.strategy, data_dir=args.data_dir, model_dir=args.model_dir, augment=True,
         load_model=False, model_name="sunrgb20200517T1349\mask_rcnn_sunrgb_0052.h5", init_epoch=1)
