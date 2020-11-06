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
import timeit

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
import pandas as pd


# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def evaluate_pr_curve(dataset_val, inference_config, model):
    # Running on 10 images. Increase for better accuracy.
    image_ids = dataset_val.image_ids
    print(len(dataset_val.image_ids))
    APs = []
    F1s = []
    ARs = []
    runtimes = []
    i = 0
    tps = 0
    fps = 0
    fns = 0
    precisions = []
    recalls = []
    tp_fp_fns = pd.DataFrame(columns=["tp", "fp", "fn", "score"])
    for image_id in image_ids:
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset_val, inference_config,
                                   image_id, use_mini_mask=False)
        molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
        # Run object detection
        start_time = timeit.timeit()
        results = model.detect([image], verbose=0)
        end_time = timeit.timeit()
        runtimes.append(end_time - start_time)
        r = results[0]
        # Compute AP

        _, _, _, tp_fp_fn = \
            utils.compute_matches(gt_bbox, gt_class_id, gt_mask,
                                  r["rois"], r["class_ids"], r["scores"], r['masks'])
        # print(overlaps)
        # print("---")
        tp_fp_fns = tp_fp_fns.append(tp_fp_fn, ignore_index=True)
        """
        tps = tps + tp
        fps = fps + fp
        fns = fns + fn
        precision = tps / (tps + fps)
        recall = tps / (tps + fns)
        print(tp, fp, fn, tps, fps, fns, precision, recall)
        
        precisions.append(precision)
        recalls.append(recall)
        """

        if i % 100 == 0:
            print(i)
        if i > 1000:
            break
        i += 1
    tp_fp_fns = tp_fp_fns.sort_values(by="score", ascending=False)
    print(tp_fp_fns)
    tp_fp_fns["acc tp"] = tp_fp_fns["tp"].expanding(1).sum()
    tp_fp_fns["acc fp"] = tp_fp_fns["fp"].expanding(1).sum()
    tp_fp_fns["acc fn"] = tp_fp_fns["fn"].expanding(1).sum()
    print(tp_fp_fns)
    # precision = tp / (tp + fp)
    # recall = tp / (tp + fn)

    tp_fp_fns["precision"] = tp_fp_fns["acc tp"] / (tp_fp_fns["acc tp"] + tp_fp_fns["acc fp"])
    tp_fp_fns["recall"] = tp_fp_fns["acc tp"] / (tp_fp_fns["acc tp"] + tp_fp_fns["acc fn"])
    print(tp_fp_fns)

    tp_fp_fns.to_csv(f"precision_recalls_{args.data_set}_{args.strategy}.csv")
    print(len(APs), len(F1s))
    np.save(f"runtimes_CPU_{args.data_set}_{args.strategy}.npy", np.array(runtimes))
    return np.mean(APs), np.mean(F1s), np.mean(ARs)


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
    ARs = []
    runtimes = []
    i = 0
    for image_id in image_ids:
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset_val, inference_config,
                                   image_id, use_mini_mask=False)
        molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
        # Run object detection
        start_time = timeit.timeit()
        results = model.detect([image], verbose=0)
        end_time = timeit.timeit()
        runtimes.append(end_time - start_time)
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
        ARs.append(recall)
        # plot_precision_recall_curve(precisions, recalls)
        data = [[precisions[i], recalls[i]] for i in range(0, len(recalls))]
        if i % 100 == 0:
            print(i)
        if i > 1000:
            break
        i += 1
    print(len(APs), len(F1s))
    np.save(f"runtimes_CPU_{args.data_set}_{args.strategy}.npy", np.array(runtimes))
    return np.mean(APs), np.mean(F1s), np.mean(ARs)


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
    config.DETECTION_MIN_CONFIDENCE = 0.8
    config.TRAIN_ROIS_PER_IMAGE = 50
    # config.IMAGE_MIN_DIM = 256
    # config.IMAGE_MAX_DIM = 256
    #config.DROPOUT_RATE = -1
    config.NUM_FILTERS = [32, 32, 64, 128, 256]
    # config.RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    config.display()

    # dataset_tst = SunRGBDDataset()
    # dataset_tst.load_sun_rgbd(data_dir, "split/test13")
    # dataset_tst = SunD3Dataset()
    # dataset_tst.load_sun_d3(data_dir, "split/test13")
    # dataset_tst = SunRGBDataset()
    # dataset_tst.load_sun_rgb(data_dir, "split/test13")
    # dataset_tst = SunRGBDFusenetDataset()
    # dataset_tst.load_sun_rgbd_fusenet(data_dir, "split/test13")
    #dataset_tst = ElevatorRGBDataset()
    #dataset_tst.load_elevator_rgb(data_dir, "test")
    #dataset_tst = ElevatorD3Dataset()
    #dataset_tst.load_elevator_d3(data_dir, "test")
    #dataset_tst = ElevatorRGBDDataset()
    #dataset_tst.load_elevator_rgbd(data_dir, "test")
    dataset_tst = ElevatorRGBDFusenetDataset()
    dataset_tst.load_elevator_rgbd_fusenet(data_dir, "test")

    dataset_tst.prepare()

    model = modellib.MaskRCNN(mode="inference", config=config,
                              model_dir=model_dir)
    print(model.keras_model.summary())
    model.load_weights(model_name, by_name=True)
    model.epoch = 50
    mean_average_precision, f1_score, mean_average_recall = evaluate_pr_curve(dataset_val=dataset_tst,
                                                                              inference_config=config,
                                                                              model=model)
    print("mAP: ", mean_average_precision)
    print("F1s: ", f1_score)
    print("mAR: ", mean_average_recall)


if __name__ == "__main__":
    # print(tf.test.is_gpu_available())
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, help="Data directory",
                        default=os.path.join(ROOT_DIR, "datasets", "elevator", "preprocessed"))
    # os.path.abspath("D:\\Data\\sun_rgbd\\crop\\"))
    parser.add_argument("-m", "--model_dir", type=str, help="Directory to store weights and results",
                        default=os.path.join(ROOT_DIR, "logs"))
    parser.add_argument("-s", "--strategy", type=str, help="[D3, RGB, RGBD, RGBDParallel, RGBDFusenet]",
                        default="RGBDFusenet")
    parser.add_argument("-w", "--data_set", type=str, help="[SUN, ELEVATOR]", default="ELEVATOR")
    args = parser.parse_args()

    main(data_dir=args.data_dir, model_dir=args.model_dir,
         model_name=args.model_dir + "/weights/mask_rcnn_elevator_rgbd_fusenet_0050.h5",
         backbone="fusenet", strategy=args.strategy, data_set=args.data_set)
