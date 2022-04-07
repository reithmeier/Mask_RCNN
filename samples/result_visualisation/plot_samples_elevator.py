# **********************************************************************************************************************
#
# brief:    simple script to plot runtimes
#
# author:   Lukas Reithmeier
# date:     15.08.2020
#
# **********************************************************************************************************************


import os
import sys
import random

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
print(ROOT_DIR)
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
import mrcnn.model as modellib

from samples.elevator import elevator_d3, elevator_rgb, elevator_rgbd, elevator_rgbd_fusenet

ELEVATOR_DIR = os.path.join(ROOT_DIR, "datasets", "../elevator", "preprocessed")
MODEL_DIR = os.path.join(ROOT_DIR, "logs/")

model_file_rgb = os.path.join(MODEL_DIR, "weights/mask_rcnn_elevator_rgb_0050.h5")
model_file_d3 = os.path.join(MODEL_DIR, "weights/mask_rcnn_elevator_d3_0050.h5")
model_file_rgbd = os.path.join(MODEL_DIR, "weights/mask_rcnn_elevator_rgbd_0050.h5")
model_file_rgbd_fusenet = os.path.join(MODEL_DIR, "weights/mask_rcnn_elevator_rgbd_fusenet_0050.h5")

config_rgb = elevator_rgb.ElevatorRGBConfig()
config_rgb.BACKBONE = "resnet50"
config_rgb.DROPOUT_RATE = -1
config_rgb.DETECTION_MIN_CONFIDENCE = 0.6
config_rgb.TRAIN_ROIS_PER_IMAGE = 100
config_rgb.BATCH_SIZE = 1
config_rgb.IMAGES_PER_GPU = 1

config_d3 = elevator_d3.ElevatorD3Config()
config_d3.BACKBONE = "resnet50"
config_d3.DROPOUT_RATE = -1
config_d3.DETECTION_MIN_CONFIDENCE = 0.7
config_d3.TRAIN_ROIS_PER_IMAGE = 200
config_d3.BATCH_SIZE = 1
config_d3.IMAGES_PER_GPU = 1

config_rgbd = elevator_rgbd.ElevatorRGBDConfig()
config_rgbd.BACKBONE = "resnet50"
config_rgbd.DROPOUT_RATE = -1
config_rgbd.DETECTION_MIN_CONFIDENCE = 0.8
config_rgbd.TRAIN_ROIS_PER_IMAGE = 50
config_rgbd.BATCH_SIZE = 1
config_rgbd.IMAGES_PER_GPU = 1

config_rgbd_fusenet = elevator_rgbd_fusenet.ElevatorRGBDFusenetConfig()
config_rgbd_fusenet.BACKBONE = "fusenet"
config_rgbd_fusenet.DETECTION_MIN_CONFIDENCE = 0.8
config_rgbd_fusenet.TRAIN_ROIS_PER_IMAGE = 50
config_rgbd_fusenet.BATCH_SIZE = 1
config_rgbd_fusenet.IMAGES_PER_GPU = 1
config_rgbd_fusenet.NUM_FILTERS = [32, 32, 64, 128, 256]

dataset_rgb = elevator_rgb.ElevatorRGBDataset()
dataset_rgb.load_elevator_rgb(ELEVATOR_DIR, "test")
dataset_rgb.prepare()
dataset_d3 = elevator_d3.ElevatorD3Dataset()
dataset_d3.load_elevator_d3(ELEVATOR_DIR, "test")
dataset_d3.prepare()
dataset_rgbd = elevator_rgbd.ElevatorRGBDDataset()
dataset_rgbd.load_elevator_rgbd(ELEVATOR_DIR, "test")
dataset_rgbd.prepare()
dataset_rgbd_fusenet = elevator_rgbd_fusenet.ElevatorRGBDFusenetDataset()
dataset_rgbd_fusenet.load_elevator_rgbd_fusenet(ELEVATOR_DIR, "test")
dataset_rgbd_fusenet.prepare()

model_rgb = modellib.MaskRCNN(mode="inference", config=config_rgb, model_dir=MODEL_DIR)
model_rgb.load_weights(model_file_rgb, by_name=True)

model_d3 = modellib.MaskRCNN(mode="inference", config=config_d3, model_dir=MODEL_DIR)
model_d3.load_weights(model_file_d3, by_name=True)

model_rgbd = modellib.MaskRCNN(mode="inference", config=config_rgbd, model_dir=MODEL_DIR)
model_rgbd.load_weights(model_file_rgbd, by_name=True)

model_rgbd_fusenet = modellib.MaskRCNN(mode="inference", config=config_rgbd_fusenet, model_dir=MODEL_DIR)
model_rgbd_fusenet.load_weights(model_file_rgbd_fusenet, by_name=True)


def plot_inference(model, dataset, image_id, rgb_image):
    image = dataset.load_image(image_id)
    results = model.detect([image], verbose=1)

    r = results[0]
    return visualize.display_instances(rgb_image, r['rois'], r['masks'], r['class_ids'],
                                       dataset.class_names, r['scores'])


def plot_sun_rgb():
    image_ids = random.choices(dataset_rgb.image_ids, k=10)

    for image_id in image_ids:
        print(image_id)

        image = dataset_rgb.load_image(image_id)
        mask, class_ids = dataset_rgb.load_mask(image_id)
        bbox = utils.extract_bboxes(mask)
        _, ground_truth = visualize.display_instances(image, bbox, mask, class_ids, dataset_rgb.class_names)

        _, result_rgb = plot_inference(model_rgb, dataset_rgb, image_id, image)
        _, result_d3 = plot_inference(model_d3, dataset_d3, image_id, image)
        _, result_rgbd = plot_inference(model_rgbd, dataset_rgbd, image_id, image)
        _, result_rgbd_fusenet = plot_inference(model_rgbd_fusenet, dataset_rgbd_fusenet, image_id, image)

        ground_truth.savefig("inference_" + str(image_id) + "_ground_truth.png")
        result_rgb.savefig("inference_" + str(image_id) + "_elevator_rgb.png")
        result_d3.savefig("inference_" + str(image_id) + "_elevator_d3.png")
        result_rgbd.savefig("inference_" + str(image_id) + "_elevator_rgbd.png")
        result_rgbd_fusenet.savefig("inference_" + str(image_id) + "_elevator_rgbd_fusenet.png")


plot_sun_rgb()
