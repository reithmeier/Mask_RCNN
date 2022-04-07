# **********************************************************************************************************************
#
# brief:    simple script to plot runtimes
#
# author:   Lukas Reithmeier
# date:     14.08.2020
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

from samples.sun import sunrgb, sund3, sunrgbd, sunrgbd_fusenet

SUN_DIR = "D:/Data/sun_rgbd/resized/"
MODEL_DIR = os.path.join(ROOT_DIR, "logs/")
print(MODEL_DIR)

model_file_rgb = os.path.join(MODEL_DIR, "weights/mask_rcnn_sunrgb_0050.h5")
model_file_d3 = os.path.join(MODEL_DIR, "weights/mask_rcnn_sund3_0050.h5")
model_file_rgbd = os.path.join(MODEL_DIR, "weights/mask_rcnn_sunrgbd_0050.h5")
model_file_rgbd_fusenet = os.path.join(MODEL_DIR, "weights/mask_rcnn_sunrgbd_fusenet_0050.h5")

config_rgb = sunrgb.SunRGBConfig()
config_rgb.BACKBONE = "resnet50"
config_rgb.DROPOUT_RATE = -1
config_rgb.DETECTION_MIN_CONFIDENCE = 0.6
config_rgb.TRAIN_ROIS_PER_IMAGE = 100
config_rgb.BATCH_SIZE = 1
config_rgb.IMAGES_PER_GPU = 1

config_d3 = sund3.SunD3Config()
config_d3.BACKBONE = "resnet50"
config_d3.DROPOUT_RATE = -1
config_d3.DETECTION_MIN_CONFIDENCE = 0.7
config_d3.TRAIN_ROIS_PER_IMAGE = 200
config_d3.BATCH_SIZE = 1
config_d3.IMAGES_PER_GPU = 1

config_rgbd = sunrgbd.SunRGBDConfig()
config_rgbd.BACKBONE = "resnet50"
config_rgbd.DROPOUT_RATE = -1
config_rgbd.DETECTION_MIN_CONFIDENCE = 0.8
config_rgbd.TRAIN_ROIS_PER_IMAGE = 50
config_rgbd.BATCH_SIZE = 1
config_rgbd.IMAGES_PER_GPU = 1

config_rgbd_fusenet = sunrgbd_fusenet.SunRGBDFusenetConfig()
config_rgbd_fusenet.DETECTION_MIN_CONFIDENCE = 0.8
config_rgbd_fusenet.TRAIN_ROIS_PER_IMAGE = 50
config_rgbd_fusenet.BATCH_SIZE = 1
config_rgbd_fusenet.IMAGES_PER_GPU = 1
config_rgbd_fusenet.NUM_FILTERS = [32, 32, 64, 128, 256]

dataset_rgb = sunrgb.SunRGBDataset()
dataset_rgb.load_sun_rgb(SUN_DIR, "split/test13")
dataset_rgb.prepare()
dataset_d3 = sund3.SunD3Dataset()
dataset_d3.load_sun_d3(SUN_DIR, "split/test13")
dataset_d3.prepare()
dataset_rgbd = sunrgbd.SunRGBDDataset()
dataset_rgbd.load_sun_rgbd(SUN_DIR, "split/test13")
dataset_rgbd.prepare()
dataset_rgbd_fusenet = sunrgbd_fusenet.SunRGBDFusenetDataset()
dataset_rgbd_fusenet.load_sun_rgbd_fusenet(SUN_DIR, "split/test13")
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

        _,result_rgb = plot_inference(model_rgb, dataset_rgb, image_id, image)
        _,result_d3 = plot_inference(model_d3, dataset_d3, image_id, image)
        _,result_rgbd = plot_inference(model_rgbd, dataset_rgbd, image_id, image)
        _,result_rgbd_fusenet = plot_inference(model_rgbd_fusenet, dataset_rgbd_fusenet, image_id, image)

        ground_truth.savefig("inference_" + str(image_id) + "_ground_truth.png")
        result_rgb.savefig("inference_" + str(image_id) + "_sun_rgb.png")
        result_d3.savefig("inference_" + str(image_id) + "_sun_d3.png")
        result_rgbd.savefig("inference_" + str(image_id) + "_sun_rgbd.png")
        result_rgbd_fusenet.savefig("inference_" + str(image_id) + "_sun_rgbd_fusenet.png")


plot_sun_rgb()
