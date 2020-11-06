# **********************************************************************************************************************
#
# brief:    script to visualize the model
#
# author:   Lukas Reithmeier
# date:     16.06.2020
#
# **********************************************************************************************************************

import argparse
import os
import sys

import cv2
import numpy as np
import tensorflow as tf

from samples.sun.sund3 import SunD3Config, SunD3Dataset
from samples.sun.sunrgb import SunRGBConfig, SunRGBDataset
from samples.sun.sunrgbd import SunRGBDConfig, SunRGBDDataset
from samples.sun.sunrgbd_fusenet import SunRGBDFusenetConfig, SunRGBDFusenetDataset
from samples.sun.sunrgbd_parallel import SunRGBDParallelConfig, SunRGBDParallelDataset

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append('.')
sys.path.append(ROOT_DIR)  # To find local version of the library
import mrcnn.model as modellib
from mrcnn.visualize import display_images

from samples.elevator.elevator_d3 import ElevatorD3Config, ElevatorD3Dataset
from samples.elevator.elevator_rgb import ElevatorRGBConfig, ElevatorRGBDataset
from samples.elevator.elevator_rgbd import ElevatorRGBDConfig, ElevatorRGBDDataset
from samples.elevator.elevator_rgbd_parallel import ElevatorRGBDParallelConfig, ElevatorRGBDParallelDataset
from samples.elevator.elevator_rgbd_fusenet import ElevatorRGBDFusenetConfig, ElevatorRGBDFusenetDataset
import skimage.io
import matplotlib.pyplot as plt


def visualize_filters(config, model_dir, model_path):
    model = modellib.MaskRCNN(mode="inference",
                              config=config,
                              model_dir=model_dir)
    model.load_weights(model_path, by_name=True)
    print(model.keras_model.summary())

    for layer in model.keras_model.layers:
        # if 'conv' not in layer.name:
        #   continue

        if not ((('res' in layer.name) and ('branch' in layer.name) and layer.name[-1] == 'b')
                or 'conv1' == layer.name):
            continue
        print(layer.name)

        filters, biases = layer.get_weights()
        f_min, f_max = np.amin(filters), np.amax(filters)

        filters = (filters - f_min) / (f_max - f_min)
        # plot first few filters
        n_filters, n_channels, ix = min(filters.shape[3], 5), min(filters.shape[2], 5), 1

        fig = plt.figure()

        for i in range(n_filters):
            # get the filter
            f = filters[:, :, :, i]

            # plot each channel separately

            for channel in range(n_channels):
                # specify subplot and turn of axis

                ax = fig.add_subplot(n_filters, n_channels, ix)
                ax.set_xticks([])
                ax.set_yticks([])

                # plot filter channel in grayscale
                ax.imshow(f[:, :, channel], cmap='gray')

                ix += 1
        # show the figure

        fig.suptitle(layer.name)

        plt.show()


def visualize_activations(config, model_dir, model_path, dataset_val):
    model = modellib.MaskRCNN(mode="inference",
                              config=config,
                              model_dir=model_dir)
    model.load_weights(model_path, by_name=True)
    print(model.keras_model.summary())
    i = 0
    image_ids = np.random.choice(dataset_val.image_ids, 10)
    for image_id in image_ids:
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset_val, config,
                                   image_id, use_mini_mask=False)
        """
        activations = model.run_graph([image], [
            ("input_image", tf.identity(model.keras_model.get_layer("input_image").output)),
            ("res2c_out", model.keras_model.get_layer("res2c_out").output),
            ("res3c_out", model.keras_model.get_layer("res3c_out").output),
            ("res4c_out", model.keras_model.get_layer("res4c_out").output),
            ("res5c_out", model.keras_model.get_layer("res5c_out").output),
            ("rpn_bbox", model.keras_model.get_layer("rpn_bbox").output),
            ("roi", model.keras_model.get_layer("ROI").output),
        ])
        """

        activations = model.run_graph([image], [
            ("input_image", tf.identity(model.keras_model.get_layer("input_image").output)),
            ("relu3fsn_branchrgb", model.keras_model.get_layer("relu3fsn_branchrgb").output),
            ("relu4fsn_branchrgb", model.keras_model.get_layer("relu4fsn_branchrgb").output),
            ("relu5cbr3_branchdpt", model.keras_model.get_layer("relu5cbr3_branchdpt").output),
            ("pl5pool_branchrgb", model.keras_model.get_layer("pl5pool_branchrgb").output),
            ("rpn_bbox", model.keras_model.get_layer("rpn_bbox").output),
            ("roi", model.keras_model.get_layer("ROI").output),
        ])

        # Backbone feature map

        display_images(np.transpose(activations["relu3fsn_branchrgb"][0, :, :, :4], [2, 0, 1]), cols=4)
        display_images(np.transpose(activations["relu4fsn_branchrgb"][0, :, :, :4], [2, 0, 1]), cols=4)
        display_images(np.transpose(activations["relu5cbr3_branchdpt"][0, :, :, :4], [2, 0, 1]), cols=4)
        display_images(np.transpose(activations["pl5pool_branchrgb"][0, :, :, :4], [2, 0, 1]), cols=4)

        activation = activations["relu4fsn_branchrgb"][0, :, :, 3]
        print(activation.shape)
        rgb_image = image[:, :, 0:3]
        dpt_image = image[:, :, 3]
        resized = cv2.resize(activation, (512, 512))
        cv2.normalize(resized, resized, 255, 0, cv2.NORM_MINMAX)
        resized = np.uint8(resized)
        resized_activation = cv2.applyColorMap(resized, cv2.COLORMAP_WINTER)

        skimage.io.imsave("sun_" + str(i) + "_act.png", resized_activation)
        skimage.io.imsave("sun_" + str(i) + "_rgb.png", rgb_image)
        skimage.io.imsave("sun_" + str(i) + "_dpt.png", dpt_image)

        cv2.waitKey(0)
        i += 1


def highlight(image, activations):
    highlighted_arr = []
    for activation in activations:

        resized = cv2.resize(activation, (512, 512))
        cv2.normalize(resized, resized, 255, 0, cv2.NORM_MINMAX)
        resized = np.uint8(resized)
        resized_rgb = cv2.applyColorMap(resized, cv2.COLORMAP_AUTUMN)

        highlight = image.copy()
        for r in range(0, highlight.shape[0]):
            for c in range(0, highlight.shape[1]):
                if resized[r, c] > 1:
                    highlight[r, c] = resized_rgb[r, c]
        alpha = 0.5
        highlighted = image.copy()
        cv2.addWeighted(image, alpha, highlight, 1 - alpha, 0, highlighted)
        highlighted_arr.append(highlighted)
    return np.array(highlighted_arr)


def main(data_set, strategy, data_dir, model_dir, model_name, backbone):
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
    config.BATCH_SIZE = 1
    config.IMAGES_PER_GPU = 1
    config.NUM_FILTERS = [32, 32, 64, 128, 256]
    config.DETECTION_MIN_CONFIDENCE = 0.8
    config.TRAIN_ROIS_PER_IMAGE = 50
    config.DROPOUT_RATE = -1
    config.display()

    visualize_filters(config=config, model_dir=model_dir, model_path=model_name)

    # visualize_activations(config=config, model_dir=model_dir, model_path=model_name, dataset_val=dataset_val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, help="Data directory",
                        default=os.path.abspath("D:/Data/sun_rgbd/crop"))
    # ROOT_DIR,
    # "datasets\\elevator\\preprocessed"))  # os.path.abspath("I:\Data\elevator\preprocessed"))
    parser.add_argument("-m", "--model_dir", type=str, help="Directory to store weights and results",
                        default=ROOT_DIR + "/logs/")
    parser.add_argument("-s", "--strategy", type=str, help="[D3, RGB, RGBD, RGBDParallel, RGBDFusenet]",
                        default="RGB")
    parser.add_argument("-w", "--data_set", type=str, help="[SUN, ELEVATOR]", default="SUN")
    args = parser.parse_args()

    main(data_set=args.data_set, strategy=args.strategy, data_dir=args.data_dir, model_dir=args.model_dir,
         model_name=os.path.join(ROOT_DIR, "logs/weights/mask_rcnn_sunrgb_0050.h5"),
         #"D:/train_logs/sun/sunrgbd_fusenet20200716T0034/mask_rcnn_sunrgbd_fusenet_0300.h5",
         backbone="resnet50")
